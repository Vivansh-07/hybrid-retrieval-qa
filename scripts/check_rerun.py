#!/usr/bin/env python
"""Check a re-run against the committed results: is every number the same?

    python scripts/run_ablation.py --variants bm25 dense fusion --out results/_live
    python scripts/check_rerun.py results/_live

Compares the re-run's ablation.csv (every metric of every variant it ran) and
significance.json (every pairwise test) with the committed files in results/. It reads
only those small summary files, which every clone has; the per-claim run files that
scripts/compare_runs.py needs are not on GitHub. Time per claim is shown but not
compared: it varies between runs. Exits with 1 if any number differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOT_COMPARED = {"variant", "latency_ms_per_query"}


def _folder(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_ablation(folder: Path) -> dict:
    with open(folder / "ablation.csv", newline="", encoding="utf-8") as fh:
        return {row["variant"]: row for row in csv.DictReader(fh)}


def read_significance(folder: Path) -> dict:
    path = folder / "significance.json"
    return json.loads(path.read_text(encoding="utf-8"))["comparisons"] if path.exists() else {}


def check(rerun: Path, baseline: Path) -> tuple[list[str], list[str]]:
    """(lines to print, differences found)."""
    new, old = read_ablation(rerun), read_ablation(baseline)
    lines = [f"{'variant':<15}{'nDCG@10 committed':>19}{'re-run':>9}{'metrics identical':>20}"
             f"{'ms/claim committed -> re-run':>32}"]
    differences = []
    for variant, row in new.items():
        if variant not in old:
            differences.append(f"{variant}: not in the committed results")
            continue
        keys = [k for k in row if k not in NOT_COMPARED]
        diff = [k for k in keys if row[k] != old[variant].get(k)]
        differences += [f"{variant} {k}: committed {old[variant].get(k)}, re-run {row[k]}" for k in diff]
        lines.append(f"{variant:<15}{old[variant]['nDCG@10']:>19}{row['nDCG@10']:>9}"
                     f"{len(keys) - len(diff):>14} of {len(keys):<3}"
                     f"{old[variant]['latency_ms_per_query']:>20} -> {row['latency_ms_per_query']}")

    new_sig, old_sig = read_significance(rerun), read_significance(baseline)
    same = [name for name in new_sig if new_sig[name] == old_sig.get(name)]
    differences += [f"significance '{name}' differs" for name in new_sig if name not in same]
    lines.append(f"significance tests identical: {len(same)} of {len(new_sig)}")
    return lines, differences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rerun", help="folder of the re-run, e.g. results/_live")
    ap.add_argument("--baseline", default="results", help="committed results to compare with (default: results)")
    args = ap.parse_args()

    lines, differences = check(_folder(args.rerun), _folder(args.baseline))
    print(f"\nRe-run {args.rerun} compared with the committed {args.baseline}:\n")
    print("\n".join(lines))
    if differences:
        print("\nDIFFERENT:\n  " + "\n  ".join(differences))
        sys.exit(1)
    print("\nEvery metric and every significance test is identical to the committed results.")


if __name__ == "__main__":
    main()
