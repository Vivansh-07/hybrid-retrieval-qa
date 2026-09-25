"""Unit tests for the retrieval metrics — worked by hand so the numbers are checkable."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.metrics import (  # noqa: E402
    evaluate_run, ndcg_at_k, paired_bootstrap, precision_at_k, recall_at_k, reciprocal_rank,
)

RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2": 1, "d5": 1, "d9": 1}  # d9 is relevant but never retrieved


def test_recall():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 1 / 3
    assert recall_at_k(RANKED, RELEVANT, 5) == 2 / 3
    assert recall_at_k(RANKED, {}, 5) == 0.0


def test_precision():
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 5) == 2 / 5
    assert precision_at_k(RANKED, RELEVANT, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5           # first hit at rank 2
    assert reciprocal_rank(["d7", "d8"], RELEVANT) == 0.0     # no hit
    assert reciprocal_rank(RANKED, RELEVANT, k=1) == 0.0      # hit is outside the cutoff


def test_ndcg_hand_computed():
    # gains at ranks 1..5: 0, 1, 0, 0, 1
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    # ideal: three relevant docs at ranks 1, 2, 3
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(RANKED, RELEVANT, 5) - dcg / idcg) < 1e-12


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 2) == 1.0
    assert ndcg_at_k(["x"], {}, 5) == 0.0


def test_evaluate_run_macro_average():
    run = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    qrels = {"q1": {"d1": 1}, "q2": {"d4": 1}}
    out = evaluate_run(run, qrels, k_values=[1, 2])
    assert out["Recall@2"] == 1.0
    assert out["Recall@1"] == 0.5          # q1 hits at rank 1, q2 does not
    assert out["MRR"] == (1.0 + 0.5) / 2
    assert out["num_queries"] == 2


def test_bootstrap_identical_runs_not_significant():
    a = {"q1": 0.5, "q2": 0.5, "q3": 0.5}
    assert paired_bootstrap(a, dict(a), n_samples=500) == 1.0
