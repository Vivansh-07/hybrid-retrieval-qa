#!/usr/bin/env python
"""Recompute significance.json from saved per-query results, without re-running retrieval.

    python scripts/significance.py
    python scripts/significance.py --results "results/set_chunking.strategy=none"

Reads <results>/runs/{bm25,dense,fusion,fusion_rerank}.json (written by the ablation)
and tests every pair of variants with the paired bootstrap. Takes seconds, so an
extra comparison never needs the 13-minute ablation again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.significance import load_runs, save_significance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--results", default="results", help="folder containing runs/ (default: results)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = Path(args.results)
    results = results if results.is_absolute() else REPO / results

    per_query = {variant: saved["per_query"] for variant, saved in load_runs(results).items()}
    if len(per_query) < 2:
        sys.exit(f"need run files for at least two variants in {results / 'runs'}; "
                 "run scripts/run_ablation.py first")

    key = f"nDCG@{cfg.get_path('evaluation.primary_k', 10)}"
    save_significance(per_query, results, key=key, seed=cfg.get_path("evaluation.seed", 42))
    print(f"\nwritten to {results / 'significance.json'}")


if __name__ == "__main__":
    main()
