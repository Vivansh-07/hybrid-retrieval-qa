#!/usr/bin/env python
"""Compare a one-change experiment against the main results, variant by variant.

    python scripts/compare_runs.py "results/set_chunking.strategy=none"
    python scripts/compare_runs.py "results/set_fusion.method=nsf" --baseline results

For every variant present in both folders it reports the mean nDCG@10 in each, the
difference, a paired bootstrap p-value and how many queries got better / worse / stayed
the same. Writes <folder>/comparison.json. Takes seconds: it only reads the saved
per-query files (runs/*.json), it does not retrieve anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.significance import compare_result_dirs, format_p  # noqa: E402


def in_repo(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="results folder of the one-change run, e.g. results/set_chunking.strategy=none")
    ap.add_argument("--baseline", default="results", help="folder to compare against (default: results)")
    ap.add_argument("--config", default="configs/scifact.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    key = f"nDCG@{cfg.get_path('evaluation.primary_k', 10)}"
    baseline, other = in_repo(args.baseline), in_repo(args.folder)

    rows = compare_result_dirs(baseline, other, key=key, seed=cfg.get_path("evaluation.seed", 42))
    if not rows:
        sys.exit(f"no variant has run files in both {baseline / 'runs'} and {other / 'runs'}")

    print(f"\nbaseline: {baseline}\nother:    {other}\n")
    print(f"{'variant':<15}{'baseline':>9}{'other':>9}{'delta':>9}  {'p':<9} better/worse/same   docs per query")
    for variant, r in rows.items():
        verdict = "" if r["p_value"] < 0.05 else "  (not significant)"
        print(f"{variant:<15}{r['baseline']:>9.4f}{r['other']:>9.4f}{r['delta']:>+9.4f}  "
              f"{format_p(r['p_value']):<9} {r['better']:>4}/{r['worse']}/{r['same']:<10} "
              f"{r['baseline_docs_per_query']} -> {r['other_docs_per_query']}{verdict}")

    def label(path: Path) -> str:  # repo-relative when possible, so the file is portable
        return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)

    payload = {"metric": key, "baseline": label(baseline), "other": label(other), "variants": rows}
    (other / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {other / 'comparison.json'}")


if __name__ == "__main__":
    main()
