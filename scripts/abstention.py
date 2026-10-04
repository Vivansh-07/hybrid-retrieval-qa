#!/usr/bin/env python
"""Could the reader tell when to say "not enough evidence"? A retrieval-confidence check.

    python scripts/abstention.py            # main configuration
    python scripts/abstention.py --best     # the best system (fine-tuned encoder, BM25 weight 0.3)

Neither reader abstains when the gold abstract is missing from its passages (Answer
quality in the paper). This asks whether a retrieval signal could trigger it. For every
claim it records two signals of the full pipeline, the top cross-encoder score and the
top dense cosine similarity, and whether any gold abstract is among the passages the
reader is given (answer.n_passages). It reports:

  * each signal's ROC AUC on the 300 test claims (0.5 = no better than chance), and
  * a threshold chosen on the 120 dev claims (the ones held out from fine-tuning; it
    maximises caught misses minus wrongly refused hits), applied once to the test claims.

Writes results/abstention.json (or results/abstention_best.json with --best).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.abstention import apply_threshold, auc, best_threshold  # noqa: E402
from hybridqa.config import apply_overrides, load_config  # noqa: E402
from hybridqa.data import dev_split, load_dataset_for  # noqa: E402
from hybridqa.pipeline import RetrievalPipeline  # noqa: E402

BEST = ["dense.model=data/models/all-MiniLM-L6-v2-scifact", "fusion.weights=[0.3, 1.0]"]
SIGNALS = ("cross_encoder_top", "dense_top")


def collect(pipe: RetrievalPipeline, queries: list[dict], qrels: dict, n_passages: int) -> list[dict]:
    rows = []
    for q in queries:
        gold = {d for d, rel in qrels.get(q["query_id"], {}).items() if rel > 0}
        ranked = pipe.retrieve(q["text"], "fusion+rerank")
        given = {cid.split("#", 1)[0] for cid, _ in ranked[:n_passages]}
        rows.append({
            "query_id": q["query_id"],
            "gold_in_passages": bool(gold & given),
            "cross_encoder_top": ranked[0][1],
            "dense_top": pipe.retrieve(q["text"], "dense")[0][1],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--best", action="store_true", help="the best system's retrievers")
    ap.add_argument("--dev-queries", type=int, default=120, help="must match finetune_dense.py")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(REPO / args.config), BEST if args.best else [])
    pipe = RetrievalPipeline(cfg).prepare()
    n = cfg.get_path("answer.n_passages", 5)
    _corpus, train_queries, train_qrels = load_dataset_for(apply_overrides(cfg, ["dataset.split=train"]))
    _, dev = dev_split(train_queries, args.dev_queries, cfg.get_path("evaluation.seed", 42))

    dev_rows = collect(pipe, dev, train_qrels, n)
    test_rows = collect(pipe, pipe.queries, pipe.qrels, n)

    report = {"config": "best" if args.best else "main", "n_passages": n,
              "dev_claims": len(dev_rows), "test_claims": len(test_rows), "signals": {}}
    print(f"\ngold abstract among the {n} passages: test {sum(r['gold_in_passages'] for r in test_rows)}"
          f"/{len(test_rows)}, dev {sum(r['gold_in_passages'] for r in dev_rows)}/{len(dev_rows)}")
    for signal in SIGNALS:
        pos = [r[signal] for r in test_rows if r["gold_in_passages"]]
        neg = [r[signal] for r in test_rows if not r["gold_in_passages"]]
        test = apply_threshold(test_rows, signal, best_threshold(dev_rows, signal))
        report["signals"][signal] = {"test_auc": round(auc(pos, neg), 3), "threshold_from_dev_on_test": test}
        print(f"{signal:>18}: test AUC {auc(pos, neg):.3f}; dev threshold {test['threshold']} on test: "
              f"abstains on {test['misses_caught']}/{test['misses']} misses and "
              f"{test['hits_refused']}/{test['hits']} hits")

    out = REPO / "results" / ("abstention_best.json" if args.best else "abstention.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
