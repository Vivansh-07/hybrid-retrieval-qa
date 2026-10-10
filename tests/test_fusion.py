"""Unit tests for RRF and the chunking / id-mapping helpers."""

import sys
from pathlib import Path

import pytest

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


def test_document_level_fusion_adds_up_votes_for_different_passages_of_one_paper():
    # The case of claim 1259: both retrievers rank the gold paper A first, but through
    # different passages; paper B has the same passage at rank 2 in both lists.
    from hybridqa.config import load_config
    from hybridqa.fusion import fuse

    bm25 = [("A#1", 9.0), ("B#0", 8.0), ("A#6", 7.0)]
    dense = [("A#4", 0.9), ("B#0", 0.8), ("C#2", 0.7)]
    cfg = load_config()

    passage_level = [cid for cid, _ in fuse([bm25, dense], cfg)]
    assert passage_level[0] == "B#0"          # the split votes let B overtake A

    document_level = [cid for cid, _ in fuse([bm25, dense], cfg.override("fusion.level", "document"))]
    assert document_level[0] == "A#1"         # A wins, shown by its best-ranked passage
    assert [c.split("#")[0] for c in document_level] == ["A", "B", "C"]   # one entry per paper


def test_post_hoc_attribution_credits_the_passage_the_sentence_came_from():
    from hybridqa.answer import attribute

    passages = ["Aspirin had no effect on cancer incidence.",
                "Daily vitamin D supplementation reduced acute respiratory infections.",
                "Vitamin D levels were measured in winter."]
    assert attribute("Vitamin D supplementation reduced respiratory infections.", passages) == 2
    assert attribute("Something unrelated entirely.", passages) == 1   # no overlap: top passage
    assert attribute("vitamin d", passages) == 2                        # tie: the higher-ranked passage


@pytest.mark.parametrize("reply, cited", [
    ("Not enough evidence.", []),          # an abstention cites nothing (issue #46)
    ("It does [2].", ["p2"]),
    ("Yes, it does.", ["p1"]),             # no marker: credited to the top passage
])
def test_cite_inline_citations(monkeypatch, reply, cited):
    from hybridqa.answer import AnswerGenerator, faithfulness_signals
    from hybridqa.config import Config

    gen = AnswerGenerator(Config({"answer": {"prompt_style": "cite_inline"}}))
    monkeypatch.setattr(gen, "_generate", lambda prompt: reply)   # no model needed
    ans = gen.answer("q", "claim", [("p1", 1.0), ("p2", 0.5)], {"p1": "first", "p2": "second"})
    assert ans.cited_chunk_ids == cited
    assert faithfulness_signals(ans)["has_citation"] == float(bool(cited))


def test_dev_split_is_deterministic_and_disjoint():
    from hybridqa.data import dev_split

    queries = [{"query_id": str(i)} for i in range(50)]
    train_a, dev_a = dev_split(queries, 10, seed=42)
    train_b, dev_b = dev_split(list(reversed(queries)), 10, seed=42)   # input order does not matter
    assert [q["query_id"] for q in dev_a] == [q["query_id"] for q in dev_b]
    assert len(dev_a) == 10 and len(train_a) == 40
    assert not {q["query_id"] for q in dev_a} & {q["query_id"] for q in train_a}


def test_short_window_drops_the_rest_unless_keep_tail():
    # issue #45: short sentences in the middle of an abstract ("Design. Setting. ...")
    long = [f"Long sentence {i} carries enough words to stand as evidence on its own." for i in range(1, 6)]
    sents = long[:3] + ["Design.", "Setting.", "Methods."] + long[3:]
    text = " ".join(sents)

    default = " ".join(c.text for c in chunk_document("d", "T", text, window=3, stride=2, min_chars=120))
    assert long[-1] not in default  # the behaviour every reported result uses

    kept = chunk_document("d", "T", text, window=3, stride=2, min_chars=120, keep_tail=True)
    assert all(s in " ".join(c.text for c in kept) for s in sents)
    assert [c.chunk_id for c in kept] == [f"d#{i}" for i in range(len(kept))]


def test_bm25_ranks_ties_in_index_order():
    # issue #48: numpy's stable argsort must give exactly the order of the old Python sort
    from hybridqa.config import Config
    from hybridqa.data import Chunk
    from hybridqa.sparse import BM25Retriever

    texts = ["aspirin cox", "aspirin cox", "statin fracture", "aspirin", "aspirin cox"]
    chunks = [Chunk(f"d{i}#0", f"d{i}", "", t) for i, t in enumerate(texts)]
    bm25 = BM25Retriever(Config({"sparse": {"top_k": 5}})).index(chunks)
    scores = bm25._bm25.get_scores(bm25._tok("aspirin cox"))
    old = [bm25.chunk_ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
    assert [cid for cid, _ in bm25.search("aspirin cox")] == old


def test_top_k_zero_means_an_empty_list_and_none_no_cut():
    runs = [[("a", 3.0), ("b", 2.0)], [("b", 0.9), ("c", 0.8)]]
    assert reciprocal_rank_fusion(runs, top_k=0) == []
    assert len(reciprocal_rank_fusion(runs, top_k=None)) == 3
    assert normalized_score_fusion(runs, top_k=0) == []
