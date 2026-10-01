#!/usr/bin/env python
"""Build the failure-analysis worksheet the paper's qualitative section needs.

    python scripts/failure_analysis.py --top-n 40           # write the worksheet
    python scripts/failure_analysis.py --summarize           # after labelling: count categories

Selects the queries where the full system still fails, plus the queries where
BM25 and dense disagree most sharply, and writes a CSV you fill in BY HAND with
one of the categories below. That manual pass is the part an examiner will
probe hardest, so do not skip it.

The worksheet shows what you need to judge each case without looking anything up:
the claim, the gold abstract (title and opening), the rank each variant gave the
gold abstract, the full system's top three abstracts, and the claim's word overlap
with its evidence. It is saved as UTF-8 with a byte-order mark so Excel opens it
with the right characters. An existing worksheet that already has labels is never
overwritten unless you pass --force.

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
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.data import load_dataset_for  # noqa: E402
from hybridqa.significance import load_runs  # noqa: E402
from hybridqa.sparse import tokenize  # noqa: E402
from hybridqa.subsets import claim_properties  # noqa: E402

CATEGORIES = [
    "terminology_mismatch", "ambiguous_query", "missing_evidence",
    "conflicting_evidence", "granularity", "reranker_error",
]
VARIANTS = ["bm25", "dense", "fusion", "fusion+rerank"]


def gold_rank(ranked_docs, gold):
    """1-based rank of the best-placed gold abstract, or '>N' if none was retrieved."""
    ranks = [i for i, d in enumerate(ranked_docs, start=1) if d in gold]
    return ranks[0] if ranks else f">{len(ranked_docs)}"


def write_worksheet(args, results: Path, out: Path) -> None:
    if out.exists() and not args.force:
        with open(out, newline="", encoding="utf-8-sig") as fh:
            if any(row.get("category", "").strip() for row in csv.DictReader(fh)):
                sys.exit(f"{out} already contains labels; not overwriting it (pass --force to replace it)")

    cfg = load_config(args.config)
    corpus, queries, qrels = load_dataset_for(cfg)
    qtext = {q["query_id"]: q["text"] for q in queries}
    runs = load_runs(results)
    missing = [v for v in VARIANTS if v not in runs]
    if missing:
        sys.exit(f"missing run files for {missing} in {results / 'runs'}; run scripts/run_ablation.py first")
    props = claim_properties(queries, corpus, qrels, tokenize)
    metric = f"Recall@{args.k}"

    def title(doc_id):
        return (corpus.get(doc_id, {}).get("title") or "").strip()

    rows = []
    for qid, scores in runs["fusion+rerank"]["per_query"].items():
        gold = set(qrels.get(qid, {}))
        recall = {v: runs[v]["per_query"].get(qid, {}).get(metric, 0.0) for v in VARIANTS}
        r = recall["fusion+rerank"]

        # a hit that fusion found but reranking lost is the most diagnostic case there is
        reranker_regression = recall["fusion"] > r
        disagreement = abs(recall["bm25"] - recall["dense"])
        priority = (1 - r) * 2 + reranker_regression * 1.5 + disagreement

        first_gold = sorted(gold)[0] if gold else ""
        top3 = runs["fusion+rerank"]["run"].get(qid, [])[:3]
        rows.append({
            "query_id": qid,
            "claim": qtext.get(qid, ""),
            "category": "",   # <- you fill this in (one of CATEGORIES)
            "notes": "",      # <- and this: one sentence on why
            "gold_doc_ids": " ".join(sorted(gold)),
            "gold_title": title(first_gold),
            "gold_abstract_start": (corpus.get(first_gold, {}).get("text") or "")[:300],
            **{f"gold_rank_{v.replace('+', '_')}": gold_rank(runs[v]["run"].get(qid, []), gold) for v in VARIANTS},
            "reranker_regression": int(reranker_regression),
            "word_overlap": props.get(qid, {}).get("overlap", ""),
            "top3_retrieved": " | ".join(f"{d}: {title(d)[:90]}" + (" (GOLD)" if d in gold else "") for d in top3),
            "_priority": priority,
        })

    rows.sort(key=lambda row: -row["_priority"])
    rows = rows[: args.top_n]
    for row in rows:
        row.pop("_priority")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel reads UTF-8 correctly
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_regress = sum(row["reranker_regression"] for row in rows)
    print(f"wrote {len(rows)} rows to {out}")
    print(f"  {n_regress} are reranker regressions (fusion found the gold doc, reranking dropped it)")
    print("  valid categories: " + ", ".join(CATEGORIES))
    print("  fill in 'category' and 'notes' by hand, then run with --summarize.")


def summarize(out: Path) -> None:
    if not out.exists():
        sys.exit(f"no worksheet at {out}; create it first")
    with open(out, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    labels = [row.get("category", "").strip() for row in rows]
    unlabelled = [row["query_id"] for row, lab in zip(rows, labels) if not lab]
    invalid = sorted({lab for lab in labels if lab and lab not in CATEGORIES})
    if invalid:
        sys.exit(f"unknown categories {invalid}; use only: {', '.join(CATEGORIES)}")

    counts = Counter(lab for lab in labels if lab)
    done = sum(counts.values())
    print(f"{done} of {len(rows)} claims labelled" + (f"; still to do: {', '.join(unlabelled)}" if unlabelled else ""))
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        print(f"  {cat:<22}{n:>4}  ({100 * n / done:.0f}%)" if done else f"  {cat:<22}{n:>4}")
    summary = {"labelled": done, "total": len(rows), "counts": {c: counts.get(c, 0) for c in CATEGORIES},
               "unlabelled": unlabelled}
    path = out.with_name("failure_summary.json")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--results", default="results", help="folder containing runs/ (default: results)")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="overwrite a worksheet that already has labels")
    ap.add_argument("--summarize", action="store_true", help="count the labelled categories")
    args = ap.parse_args()

    results = Path(args.results)
    results = results if results.is_absolute() else REPO / results
    out = results / "failure_analysis.csv"
    summarize(out) if args.summarize else write_worksheet(args, results, out)


if __name__ == "__main__":
    main()
