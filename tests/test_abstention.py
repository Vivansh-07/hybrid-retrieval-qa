"""The abstention numbers in the paper (AUC, misses caught, hits refused) rest on these helpers."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.abstention import apply_threshold, auc, best_threshold  # noqa: E402


def _rows(hits, misses, signal="s"):
    return [{"gold_in_passages": True, signal: v} for v in hits] + \
           [{"gold_in_passages": False, signal: v} for v in misses]


def test_auc_extremes_and_ties():
    assert auc([3, 4], [1, 2]) == 1.0      # every hit above every miss
    assert auc([1, 2], [3, 4]) == 0.0      # every hit below
    assert auc([1, 1], [1, 1]) == 0.5      # all tied: chance
    assert auc([2], [1, 2, 3]) == 0.5      # one win, one tie (half), one loss


def test_auc_matches_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = random.Random(0)
    pos = [round(rng.gauss(1.0, 1.0), 1) for _ in range(40)]  # rounding makes ties
    neg = [round(rng.gauss(0.0, 1.0), 1) for _ in range(25)]
    expected = sklearn_metrics.roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
    assert auc(pos, neg) == pytest.approx(expected)


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        auc([1.0], [])


def test_threshold_separates_when_possible():
    rows = _rows(hits=[5, 6, 7], misses=[1, 2])
    t = best_threshold(rows, "s")
    assert t == 5  # the lowest value that is above every miss
    result = apply_threshold(rows, "s", t)
    assert result == {"threshold": 5, "misses_caught": 2, "misses": 2, "hits_refused": 0, "hits": 3}


def test_threshold_trades_off_when_classes_overlap():
    rows = _rows(hits=[2, 6, 7, 8], misses=[1, 3])
    # t=2 catches 1/2 misses, refuses 0/4 hits -> 0.50
    # t=6 catches 2/2 misses, refuses 1/4 hits -> 0.75, the best trade-off
    assert best_threshold(rows, "s") == 6
    result = apply_threshold(rows, "s", 6)
    assert (result["misses_caught"], result["hits_refused"]) == (2, 1)


def test_threshold_needs_both_classes():
    with pytest.raises(ValueError):
        best_threshold(_rows(hits=[1, 2], misses=[]), "s")
