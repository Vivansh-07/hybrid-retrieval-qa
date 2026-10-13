"""The full paper (paper_full/) quotes the numbers in results/, and follows the course's rules.

Every number is read from results/ and rounded the way the text reports it, so rewriting a
chapter in your own words cannot change a number by accident. The course's rules are checked
too: an abstract under 200 words and 40 to 50 references.
"""

import csv
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FULL = REPO / "paper_full"


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


TEXT = "\n".join(_strip_comments(f.read_text(encoding="utf-8")) for f in sorted((FULL / "sections").glob("*.tex")))


def _ndcg(folder: str, variant: str, metric: str = "nDCG@10") -> float:
    with open(RESULTS / folder / "ablation.csv", newline="", encoding="utf-8") as fh:
        return next(float(row[metric]) for row in csv.DictReader(fh) if row["variant"] == variant)


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


NF_COMPARE = _json("nfcorpus_comparisons.json")["comparisons"]

NUMBERS = [
    # SciFact: the standard pipeline and the improvement
    ("SciFact BM25", _ndcg(".", "bm25")),
    ("SciFact dense", _ndcg(".", "dense")),
    ("SciFact fusion", _ndcg(".", "fusion")),
    ("SciFact standard pipeline", _ndcg(".", "fusion+rerank")),
    ("SciFact fine-tuned dense", _ndcg("dense_minilm_finetuned", "dense")),
    ("SciFact improved retriever", _ndcg("dense_minilm_finetuned__fusion_weighted", "fusion")),
    ("SciFact improved Recall@10", _ndcg("dense_minilm_finetuned__fusion_weighted", "fusion", "Recall@10")),
    # NFCorpus: the same recipe
    ("NFCorpus BM25", _ndcg("nfcorpus", "bm25")),
    ("NFCorpus dense", _ndcg("nfcorpus", "dense")),
    ("NFCorpus fusion", _ndcg("nfcorpus", "fusion")),
    ("NFCorpus standard pipeline", _ndcg("nfcorpus", "fusion+rerank")),
    ("NFCorpus fine-tuned dense", _ndcg("nfcorpus_finetuned__fusion_weighted", "dense")),
    ("NFCorpus improved retriever", _ndcg("nfcorpus_finetuned__fusion_weighted", "fusion")),
    ("NFCorpus improved + reranker", _ndcg("nfcorpus_finetuned__fusion_weighted__rerank", "fusion+rerank")),
    ("NFCorpus SciFact encoder", _ndcg("nfcorpus_scifact_encoder__fusion_weighted", "fusion")),
]


@pytest.mark.parametrize("what, value", NUMBERS, ids=[n[0] for n in NUMBERS])
def test_number_appears_in_full_paper(what, value):
    assert f"{value:.3f}" in TEXT, f"paper_full does not mention {what} = {value:.3f} (from results/)"


@pytest.mark.parametrize("name", [
    "improved vs standard pipeline (fusion + rerank)",
    "fine-tuned dense vs pretrained dense",
    "SciFact encoder, not retrained (w = 0.3) vs standard fusion",
    "improved + rerank vs improved",
])
def test_nfcorpus_p_values_in_full_paper(name):
    p = NF_COMPARE[name]["p_value"]
    assert f"p = {p:.2f}" in TEXT, f"paper_full should give p = {p:.2f} for '{name}'"


def test_fusion_weights_in_full_paper():
    scifact = _json("fusion_weights_all-MiniLM-L6-v2-scifact.json")["chosen"]["bm25_weight"]
    nfcorpus = _json("fusion_weights_all-MiniLM-L6-v2-nfcorpus.json")["chosen"]["bm25_weight"]
    assert f"w = {scifact}" in TEXT and f"w = {nfcorpus}" in TEXT
    for row in _json("fusion_weights_all-MiniLM-L6-v2-scifact.json")["grid"]:
        assert f"{row['nDCG@10']:.3f}" in TEXT, f"development score for w = {row['bm25_weight']} missing"


def test_abstract_is_under_200_words():
    abstract = (FULL / "sections" / "00_abstract.tex").read_text(encoding="utf-8")
    body = _strip_comments(abstract.split("\\begin{abstract}", 1)[1].split("\\medskip", 1)[0])
    words = re.sub(r"\$[^$]*\$", "X", body).replace("\\noindent", "").split()
    assert len(words) < 200, f"the abstract has {len(words)} words; the course allows fewer than 200"


def test_references_between_40_and_50():
    entries = re.findall(r"^@\w+\{", (FULL / "references.bib").read_text(encoding="utf-8"), re.M)
    assert 40 <= len(entries) <= 50, f"{len(entries)} references; the course asks for 40 to 50"
