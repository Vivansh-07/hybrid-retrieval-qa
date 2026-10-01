"""Unit tests for checking and summarising hand-scored answers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.scoring import check_row, summarize_scores  # noqa: E402


def row(item, c="", co="", f="", ca="", notes=""):
    return {"item": item, "correctness": c, "completeness": co, "faithfulness": f,
            "citation_accuracy": ca, "notes": notes}


def test_unscored_rows_are_not_errors():
    assert check_row(row(1)) == []


def test_scores_must_be_0_1_or_2():
    assert check_row(row(1, "2", "1", "2", "2")) == []
    assert any("correctness" in p for p in check_row(row(1, "3", "1", "2", "2")))


def test_blank_faithfulness_only_for_no_content_answers():
    assert check_row(row(1, "0", "0", "", "1", notes="no_content")) == []
    assert check_row(row(1, "2", "2", "", "2"))  # blank without the note is an error


def test_summary_per_reader_from_the_hidden_key():
    rows = [row(1, "2", "2", "2", "2"), row(2, "0", "0", "", "0", notes="no_content"), row(3, "1", "1", "1", "2"),
            row(4)]  # item 4 not scored yet
    key = {"1": {"reader": "generative"}, "2": {"reader": "generative"}, "3": {"reader": "extractive"},
           "4": {"reader": "extractive"}}
    s = summarize_scores(rows, key)
    assert s["generative"]["n"] == 2 and s["generative"]["no_content"] == 1
    assert s["generative"]["correctness"] == 1.0
    assert s["generative"]["faithfulness"] == 2.0  # averaged over the answer with content only
    assert s["extractive"]["n"] == 1 and s["extractive"]["citation_accuracy"] == 2.0
