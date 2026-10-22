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


def _write_run(folder, variant, per_query_scores, docs, store_name=False):
    # store_name=False writes a run file as saved before the variant name was stored in it
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{variant.replace('+', '_')}.json").write_text(json.dumps({
        **({"variant": variant} if store_name else {}),
        "run": {q: docs for q in per_query_scores},
        "per_query": per_query(per_query_scores),
    }), encoding="utf-8")


def test_compare_result_dirs_pairs_the_same_variant_across_two_configs(tmp_path):
    from hybridqa.significance import compare_result_dirs

    base, other = tmp_path / "main", tmp_path / "side"
    _write_run(base, "fusion+rerank", {"q1": 0.5, "q2": 0.5}, docs=["d1", "d2"])
    _write_run(other, "fusion+rerank", {"q1": 1.0, "q2": 0.5}, docs=["d1", "d2", "d3", "d4"])
    _write_run(base, "bm25", {"q1": 0.5}, docs=["d1"])  # not in `other`: must be skipped

    rows = compare_result_dirs(base, other, n_samples=100)
    assert list(rows) == ["fusion+rerank"]
    r = rows["fusion+rerank"]
    assert (r["baseline"], r["other"], r["delta"]) == (0.5, 0.75, 0.25)
    assert (r["better"], r["worse"], r["same"]) == (1, 0, 1)
    assert (r["baseline_docs_per_query"], r["other_docs_per_query"]) == (2.0, 4.0)


def test_load_runs_keeps_the_variant_name_stored_in_the_file(tmp_path):
    # issue #50: dense_ft.json came back as "dense+ft" when the name was guessed from the file name
    from hybridqa.significance import load_runs

    _write_run(tmp_path, "dense_ft", {"q1": 0.5}, docs=["d1"], store_name=True)
    assert list(load_runs(tmp_path)) == ["dense_ft"]


def test_load_runs_still_reads_run_files_saved_without_the_name(tmp_path):
    # issue #50: a partial re-run leaves older run files (no name inside) next to new ones
    from hybridqa.significance import load_runs

    _write_run(tmp_path, "fusion+rerank", {"q1": 0.5}, docs=["d1"])  # fusion_rerank.json, old format
    _write_run(tmp_path, "dense_ft", {"q1": 0.5}, docs=["d1"], store_name=True)
    runs = load_runs(tmp_path)
    assert sorted(runs) == ["dense_ft", "fusion+rerank"]
    assert runs["fusion+rerank"]["per_query"] == per_query({"q1": 0.5})


def test_save_results_stores_the_variant_name_for_load_runs(tmp_path):
    # issue #50: save_results writes the name into every run file and load_runs reads it back
    from hybridqa.config import load_config
    from hybridqa.pipeline import RetrievalPipeline, VariantResult
    from hybridqa.significance import load_runs

    cfg = load_config().override("paths.results_dir", str(tmp_path / "results"))
    results = {name: VariantResult(name, {"nDCG@10": 0.5}, 1.0, run={"q1": ["d1"]},
                                   per_query=per_query({"q1": 0.5}))
               for name in ("fusion+rerank", "dense_ft")}
    out = RetrievalPipeline(cfg, verbose=False).save_results(results)

    saved = json.loads((out / "runs" / "fusion_rerank.json").read_text(encoding="utf-8"))
    assert saved["variant"] == "fusion+rerank"
    runs = load_runs(out)
    assert sorted(runs) == ["dense_ft", "fusion+rerank"]
    assert runs["dense_ft"]["run"] == {"q1": ["d1"]} and runs["dense_ft"]["per_query"] == per_query({"q1": 0.5})
