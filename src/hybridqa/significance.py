"""Paired significance tests between retrieval variants.

Every headline claim in the paper compares two variants on the same 300 queries, so
each comparison here is paired: per query, how much did system B gain over A? The
bootstrap then asks how often a gain that large appears by chance.

Used by scripts/run_ablation.py after a run, and by scripts/significance.py, which
recomputes significance.json from saved run files in seconds (no retrieval rerun).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .metrics import paired_bootstrap

PerQuery = Dict[str, Dict[str, float]]  # query_id -> {metric: value}

# (baseline, system): each claim in the paper, in the order the ablation builds up
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("bm25", "dense"),            # does a general-purpose dense model beat BM25?
    ("bm25", "fusion"),           # does fusion beat each retriever alone?
    ("dense", "fusion"),
    ("bm25", "fusion+rerank"),    # full system vs each earlier stage
    ("dense", "fusion+rerank"),
    ("fusion", "fusion+rerank"),  # does the reranker add anything on top of fusion?
)


def compare_runs(
    per_query: Dict[str, PerQuery],
    pairs: Iterable[Tuple[str, str]] = PAIRS,
    key: str = "nDCG@10",
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, dict]:
    """For each (baseline, system) pair: mean gain, bootstrap p-value, and how many
    queries got better / worse / stayed the same. Pairs with a missing variant are skipped."""
    out: Dict[str, dict] = {}
    for base, system in pairs:
        if base not in per_query or system not in per_query:
            continue
        a = {q: m[key] for q, m in per_query[base].items()}
        b = {q: m[key] for q, m in per_query[system].items()}
        qids = sorted(set(a) & set(b))
        if not qids:
            continue
        out[f"{system} vs {base}"] = {
            "delta": round(sum(b[q] - a[q] for q in qids) / len(qids), 4),
            "p_value": paired_bootstrap(a, b, n_samples, seed),
            "better": sum(b[q] > a[q] for q in qids),
            "worse": sum(b[q] < a[q] for q in qids),
            "same": sum(b[q] == a[q] for q in qids),
        }
    return out


def format_p(p: float, n_samples: int = 10000) -> str:
    """A bootstrap with n resamples cannot resolve p below 1/n: report p = 0 as '< 1/n'."""
    return f"< {1 / n_samples:g}" if p == 0 else f"{p:.4f}"


def save_significance(
    per_query: Dict[str, PerQuery],
    out_dir: str | Path,
    key: str = "nDCG@10",
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Run every comparison, write <out_dir>/significance.json, print a readable table."""
    comparisons = compare_runs(per_query, PAIRS, key, n_samples, seed)
    payload = {
        "metric": key,
        "test": "two-sided paired bootstrap",
        "n_samples": n_samples,
        "seed": seed,
        "comparisons": comparisons,
    }
    Path(out_dir, "significance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== Paired bootstrap on {key} ({n_samples} resamples) ===")
    for name, s in comparisons.items():
        verdict = "significant" if s["p_value"] < 0.05 else "not significant"
        print(f"  {name:<24} delta={s['delta']:+.4f}  p={format_p(s['p_value'], n_samples):<9}"
              f"  better/worse/same = {s['better']}/{s['worse']}/{s['same']}  ({verdict})")
    return payload
