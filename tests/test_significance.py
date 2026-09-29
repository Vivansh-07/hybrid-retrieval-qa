"""Unit tests for the pairwise significance comparisons."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.significance import compare_runs, format_p, save_significance  # noqa: E402


def per_query(scores):
    return {q: {"nDCG@10": s} for q, s in scores.items()}


def test_identical_runs_show_no_difference():
    run = per_query({"q1": 0.5, "q2": 1.0, "q3": 0.0})
    result = compare_runs({"bm25": run, "dense": run}, pairs=[("bm25", "dense")])
    s = result["dense vs bm25"]
    assert s["delta"] == 0 and s["p_value"] == 1.0
    assert (s["better"], s["worse"], s["same"]) == (0, 0, 3)


def test_consistent_gain_on_every_query_is_significant():
    base = per_query({f"q{i}": 0.5 for i in range(30)})
    system = per_query({f"q{i}": 0.6 for i in range(30)})
    s = compare_runs({"bm25": base, "fusion": system}, pairs=[("bm25", "fusion")])["fusion vs bm25"]
    assert abs(s["delta"] - 0.1) < 1e-9
    assert s["p_value"] < 0.05
    assert s["better"] == 30


def test_better_worse_same_counts():
    base = per_query({"q1": 0.5, "q2": 0.5, "q3": 0.5})
    system = per_query({"q1": 1.0, "q2": 0.0, "q3": 0.5})
    s = compare_runs({"fusion": base, "fusion+rerank": system},
                     pairs=[("fusion", "fusion+rerank")])["fusion+rerank vs fusion"]
    assert (s["better"], s["worse"], s["same"]) == (1, 1, 1)


def test_pairs_with_a_missing_variant_are_skipped():
    run = per_query({"q1": 0.5})
    assert compare_runs({"bm25": run}, pairs=[("bm25", "dense")]) == {}


def test_p_value_of_zero_is_reported_as_below_resolution():
    assert format_p(0.0, 10000) == "< 0.0001"
    assert format_p(0.0253) == "0.0253"


def test_save_writes_all_default_pairs(tmp_path):
    run = per_query({"q1": 0.5, "q2": 1.0})
    variants = {v: run for v in ("bm25", "dense", "fusion", "fusion+rerank")}
    save_significance(variants, tmp_path, n_samples=100)
    saved = json.loads((tmp_path / "significance.json").read_text(encoding="utf-8"))
    assert saved["metric"] == "nDCG@10" and saved["n_samples"] == 100
    assert len(saved["comparisons"]) == 6
