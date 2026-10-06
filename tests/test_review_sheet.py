"""docs/FAILURE_REVIEW.md is regenerated whenever results/failure_analysis.csv changes."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _failure_analysis():
    spec = importlib.util.spec_from_file_location("failure_analysis", REPO / "scripts" / "failure_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_sheet_matches_the_worksheet(tmp_path):
    fresh = tmp_path / "FAILURE_REVIEW.md"
    _failure_analysis().review_sheet(REPO / "results" / "failure_analysis.csv", fresh)
    committed = (REPO / "docs" / "FAILURE_REVIEW.md").read_text(encoding="utf-8")
    assert fresh.read_text(encoding="utf-8") == committed, (
        "docs/FAILURE_REVIEW.md is out of date: run python scripts/failure_analysis.py --markdown")
