"""Unit tests for RRF and the chunking / id-mapping helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.data import chunk_document, chunk_to_doc, split_sentences  # noqa: E402
from hybridqa.fusion import (  # noqa: E402
    min_max_normalize, normalized_score_fusion, reciprocal_rank_fusion,
)


def test_rrf_scores_match_the_formula():
    run_a = [("d1", 9.0), ("d2", 4.0)]
    run_b = [("d2", 0.9), ("d3", 0.8)]
    fused = dict(reciprocal_rank_fusion([run_a, run_b], k=60))
    assert abs(fused["d1"] - 1 / 61) < 1e-12
    assert abs(fused["d2"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["d3"] - 1 / 62) < 1e-12


def test_rrf_promotes_documents_both_systems_agree_on():
    # d2 is second in both lists; d1 is first in one and absent from the other
    run_a = [("d1", 9.0), ("d2", 4.0)]
    run_b = [("d9", 0.9), ("d2", 0.8)]
    ranked = [d for d, _ in reciprocal_rank_fusion([run_a, run_b], k=60)]
    assert ranked[0] == "d2"


def test_rrf_is_scale_invariant():
    """The whole reason to fuse ranks: multiplying one system's scores changes nothing."""
    a = [("d1", 100.0), ("d2", 50.0)]
    b = [("d1", 0.001), ("d2", 0.0005)]
    assert reciprocal_rank_fusion([a]) == reciprocal_rank_fusion([b])


def test_rrf_weights_and_topk():
    a = [("d1", 1.0)]
    b = [("d2", 1.0)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60, weights=[2.0, 1.0]))
    assert fused["d1"] > fused["d2"]
    assert len(reciprocal_rank_fusion([a, b], top_k=1)) == 1


def test_min_max_normalize_handles_constant_scores():
    assert min_max_normalize([("d1", 5.0), ("d2", 5.0)]) == [("d1", 1.0), ("d2", 1.0)]
    assert min_max_normalize([]) == []


def test_normalized_score_fusion():
    a = [("d1", 10.0), ("d2", 0.0)]
    b = [("d2", 1.0), ("d1", 0.0)]
    fused = dict(normalized_score_fusion([a, b]))
    assert abs(fused["d1"] - fused["d2"]) < 1e-12  # perfectly opposed -> tie


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


def test_chunking_overlaps_and_ids():
    text = " ".join(f"Sentence number {i} is long enough to matter here." for i in range(1, 8))
    chunks = chunk_document("doc7", "Title", text, window=3, stride=2, min_chars=10)
    assert len(chunks) > 1
    assert all(c.doc_id == "doc7" for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"doc7#{i}" for i in range(len(chunks))]
    # stride < window means consecutive chunks share a sentence
    assert chunks[0].text.split(".")[-2].strip() in chunks[1].text


def test_short_document_is_one_chunk():
    chunks = chunk_document("d", "T", "Only one sentence here.", window=3)
    assert len(chunks) == 1 and chunks[0].chunk_id == "d#0"


def test_chunk_to_doc_dedupes_and_preserves_order():
    assert chunk_to_doc(["a#1", "a#0", "b#3", "a#2", "c#0"]) == ["a", "b", "c"]


def test_indexable_prepends_title():
    c = chunk_document("d", "My Title", "Body text here.", window=3)[0]
    assert c.indexable.startswith("My Title.")
