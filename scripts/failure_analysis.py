#!/usr/bin/env python
"""Build the failure-analysis worksheet the paper's qualitative section needs.

    python scripts/failure_analysis.py --top-n 40

Selects the queries where the full system still fails, plus the queries where
BM25 and dense disagree most sharply, and writes a CSV you fill in BY HAND with
one of the categories below. That manual pass is the part an examiner will
probe hardest, so do not skip it.

Categories (fix these in the paper; do not invent new ones while annotating):
  terminology_mismatch  gold uses different vocabulary than the query
  ambiguous_query       the query underspecifies what counts as an answer
  missing_evidence      no passage in the corpus supports an answer
  conflicting_evidence  passages support contradictory answers
  granularity           right document, wrong chunk retrieved
  reranker_error        gold was in the fused candidates but the reranker demoted it
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.data import load_dataset_for  # noqa: E402

CATEGORIES = [
    "terminology_mismatch", "ambiguous_query", "missing_evidence",
    "conflicting_evidence", "granularity", "reranker_error",
]


def load_run(name: str) -> dict:
    path = REPO / "results" / "runs" / f"{name.replace('+', '_')}.json"
    if not path.exists():
        sys.exit(f"missing {path} — run scripts/run_ablation.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    _corpus, queries, qrels = load_dataset_for(cfg)
    qtext = {q["query_id"]: q["text"] for q in queries}

    full = load_run("fusion+rerank")
    bm25 = load_run("bm25")
    dense = load_run("dense")
    fused = load_run("fusion")
    metric = f"Recall@{args.k}"

    rows = []
    for qid, scores in full["per_query"].items():
        gold = set(qrels.get(qid, {}))
        b = bm25["per_query"].get(qid, {}).get(metric, 0.0)
        d = dense["per_query"].get(qid, {}).get(metric, 0.0)
        f = fused["per_query"].get(qid, {}).get(metric, 0.0)
        r = scores.get(metric, 0.0)

        # a hit that fusion found but reranking lost is the most diagnostic case there is
        reranker_regression = f > r
        disagreement = abs(b - d)
        priority = (1 - r) * 2 + reranker_regression * 1.5 + disagreement

        rows.append({
            "query_id": qid,
            "query": qtext.get(qid, ""),
            "gold_doc_ids": " ".join(sorted(gold)),
            f"bm25_{metric}": round(b, 3),
            f"dense_{metric}": round(d, 3),
            f"fusion_{metric}": round(f, 3),
            f"full_{metric}": round(r, 3),
            "reranker_regression": int(reranker_regression),
            "retrieved_top5": " ".join(full["run"].get(qid, [])[:5]),
            "_priority": priority,
            "category": "",   # <- you fill this in
            "notes": "",      # <- and this
        })

    rows.sort(key=lambda r: -r["_priority"])
    rows = rows[: args.top_n]
    for r in rows:
        r.pop("_priority")

    out = REPO / "results" / "failure_analysis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_regress = sum(r["reranker_regression"] for r in rows)
    print(f"wrote {len(rows)} rows to {out}")
    print(f"  {n_regress} are reranker regressions (fusion found the gold doc, reranking dropped it)")
    print("  valid categories: " + ", ".join(CATEGORIES))
    print("  annotate the 'category' column by hand, then report the distribution in the paper.")


if __name__ == "__main__":
    main()
