"""scripts/check_rerun.py: a re-run with the same numbers passes, any changed number is reported."""

import importlib.util
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _check_rerun():
    spec = importlib.util.spec_from_file_location("check_rerun", REPO / "scripts" / "check_rerun.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rerun_of_the_committed_results(tmp_path: Path) -> Path:
    rerun = tmp_path / "_live"
    rerun.mkdir()
    for name in ("ablation.csv", "significance.json"):
        shutil.copy(REPO / "results" / name, rerun / name)
    return rerun


def test_identical_rerun_passes_even_with_different_timing(tmp_path):
    rerun = _rerun_of_the_committed_results(tmp_path)
    text = (rerun / "ablation.csv").read_text(encoding="utf-8")
    (rerun / "ablation.csv").write_text(text.replace("79.79", "109.74"), encoding="utf-8")  # timing only
    lines, differences = _check_rerun().check(rerun, REPO / "results")
    assert differences == []
    assert "significance tests identical: 6 of 6" in lines


def test_changed_metric_and_test_are_reported(tmp_path):
    rerun = _rerun_of_the_committed_results(tmp_path)
    text = (rerun / "ablation.csv").read_text(encoding="utf-8")
    (rerun / "ablation.csv").write_text(text.replace("0.6562", "0.6563", 1), encoding="utf-8")
    sig = json.loads((rerun / "significance.json").read_text(encoding="utf-8"))
    sig["comparisons"]["dense vs bm25"]["better"] += 1
    (rerun / "significance.json").write_text(json.dumps(sig), encoding="utf-8")
    _, differences = _check_rerun().check(rerun, REPO / "results")
    assert any(d.startswith("bm25 nDCG@10") for d in differences)
    assert "significance 'dense vs bm25' differs" in differences
