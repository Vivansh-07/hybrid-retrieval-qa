"""Unit tests for the query-subset analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.sparse import tokenize  # noqa: E402
from hybridqa.subsets import boolean_split, claim_properties, compare_in_groups, median_split  # noqa: E402

CORPUS = {
    "d1": {"title": "Vitamin D trial", "text": "Vitamin D reduced respiratory infections in adults."},
    "d2": {"title": "Aspirin", "text": "Aspirin had no effect on cancer."},
    "d3": {"title": "Interleukin", "text": "IL-6 levels rose after surgery."},
}
QUERIES = [
    {"query_id": "q1", "text": "Vitamin D reduces respiratory infections."},  # same words as d1
    {"query_id": "q2", "text": "Daily cholecalciferol prevents colds."},       # paraphrase of d1
    {"query_id": "q3", "text": "IL-6 increases after an operation."},           # contains a digit
]
QRELS = {"q1": {"d1": 1}, "q2": {"d1": 1}, "q3": {"d3": 1}}


def test_claim_properties():
    props = claim_properties(QUERIES, CORPUS, QRELS, tokenize)
    assert props["q3"]["has_number"] and not props["q1"]["has_number"]
    assert props["q1"]["overlap"] == 1.0      # every content term of q1 occurs in d1
    assert props["q2"]["overlap"] == 0.0      # the paraphrase shares no term with d1
    assert props["q3"]["rarest_df"] == 1      # "il-6" occurs in one abstract only
    assert props["q2"]["rarest_df"] == 0      # none of its terms occurs in the corpus


def test_splits_cover_every_claim_exactly_once():
    props = claim_properties(QUERIES, CORPUS, QRELS, tokenize)
    for groups in (boolean_split(props, "has_number", "yes", "no"), median_split(props, "overlap", "low", "high")[0]):
        ids = [q for qids in groups.values() for q in qids]
        assert sorted(ids) == ["q1", "q2", "q3"]


def test_compare_in_groups_reports_each_group_separately():
    per_query = {
        "bm25": {"q1": {"nDCG@10": 1.0}, "q2": {"nDCG@10": 0.0}, "q3": {"nDCG@10": 1.0}},
        "dense": {"q1": {"nDCG@10": 1.0}, "q2": {"nDCG@10": 1.0}, "q3": {"nDCG@10": 0.5}},
    }
    report = compare_in_groups(per_query, {"paraphrased": ["q2"], "rest": ["q1", "q3"]},
                               pairs=[("bm25", "dense")], n_samples=100)
    assert report["paraphrased"]["mean"] == {"bm25": 0.0, "dense": 1.0}
    assert report["paraphrased"]["comparisons"]["dense vs bm25"]["delta"] == 1.0
    assert report["rest"]["n"] == 2 and report["rest"]["mean"]["dense"] == 0.75
