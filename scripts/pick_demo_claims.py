#!/usr/bin/env python
"""Find good test claims to show in the live demo, from the saved runs.

    python scripts/pick_demo_claims.py                   # the best system (demo.py --best)
    python scripts/pick_demo_claims.py --results results # the main configuration (demo.py)

For every test claim it looks up the rank of the first gold abstract in each variant's
saved ranking, and lists the shortest claims (easiest to read out) in five groups:
BM25 finds it but dense does not, the reverse, fusion beats both, the reranker pushes it
out, and the reranker pulls it to the top. "Finds it" means in the top 5, which is what
the demo shows per column. Retrieves nothing, so it takes seconds; because retrieval is
deterministic, the demo shows the same ranks.

--results takes several folders; a variant in a later folder replaces the same variant
in an earlier one (the weighted-fusion run only saved fusion and fusion+rerank).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.data import load_dataset_for  # noqa: E402
from hybridqa.significance import load_runs  # noqa: E402

SHOWN = 5  # documents per column in the demo
BEST = ["results/dense_minilm_finetuned", "results/dense_minilm_finetuned__fusion_weighted"]
NEVER = 10_000  # rank of a gold abstract that is not retrieved at all

GROUPS = {
    "BM25 finds it, dense does not": lambda r: r["bm25"] == 1 and r["dense"] > SHOWN,
    "Dense finds it, BM25 does not": lambda r: r["dense"] == 1 and r["bm25"] > SHOWN,
    "Fusion beats both retrievers": lambda r: r["fusion"] == 1 and min(r["bm25"], r["dense"]) > 1,
    "The reranker pushes it out": lambda r: r["fusion"] <= 3 and r["fusion+rerank"] > SHOWN,
    "The reranker pulls it to the top": lambda r: r["fusion"] >= 3 and r["fusion+rerank"] == 1,
}


def first_gold_rank(ranking: list[str], gold: set[str]) -> int:
    return next((i for i, doc in enumerate(ranking, start=1) if doc in gold), NEVER)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--results", nargs="+", default=BEST, help="folders containing runs/")
    ap.add_argument("--per-group", type=int, default=3)
    args = ap.parse_args()

    runs = {}
    for folder in args.results:
        runs.update({v: r["run"] for v, r in load_runs(REPO / folder).items()})
    missing = {"bm25", "dense", "fusion", "fusion+rerank"} - set(runs)
    if missing:
        sys.exit(f"no saved runs for {sorted(missing)} in {args.results}")

    _corpus, queries, qrels = load_dataset_for(load_config(REPO / args.config))
    claims = []
    for q in queries:
        gold = {doc for doc, rel in qrels.get(q["query_id"], {}).items() if rel > 0}
        ranks = {v: first_gold_rank(runs[v].get(q["query_id"], []), gold) for v in runs}
        claims.append((q["query_id"], q["text"], ranks))
    claims.sort(key=lambda c: len(c[1]))

    def show(rank: int) -> str:
        return "-" if rank == NEVER else str(rank)

    print(f"Rank of the first gold abstract per variant (- = not retrieved); runs: {', '.join(args.results)}\n")
    for name, test in GROUPS.items():
        hits = [c for c in claims if test(c[2])]
        print(f"{name}: {len(hits)} claims")
        for qid, text, r in hits[: args.per_group]:
            print(f"  [{qid}] {text}\n        BM25 {show(r['bm25'])} · dense {show(r['dense'])} · "
                  f"fusion {show(r['fusion'])} · fusion+rerank {show(r['fusion+rerank'])}")
        print()


if __name__ == "__main__":
    main()
