"""The headline numbers in the paper and README match the result files they come from.

Every number is read from results/ and rounded the way the text reports it, so an edit
that changes a number in the text (or a rerun that changes a result) fails here until
the two agree again.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TEXTS = {
    "paper/main.tex": (REPO / "paper" / "main.tex").read_text(encoding="utf-8"),
    "README.md": (REPO / "README.md").read_text(encoding="utf-8"),
}


def _variant(run_dir: str, variant: str, metric: str = "nDCG@10") -> float:
    data = json.loads((RESULTS / run_dir / "metrics.json").read_text(encoding="utf-8"))
    return next(v[metric] for v in data["variants"] if v["variant"] == variant)


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# (what, value from results/, decimals, where it must appear)
RETRIEVAL = [
    ("BM25", _variant("", "bm25"), 3, TEXTS),
    ("dense (pretrained MiniLM)", _variant("", "dense"), 3, TEXTS),
    ("RRF fusion", _variant("", "fusion"), 3, TEXTS),
    ("fusion + rerank", _variant("", "fusion+rerank"), 3, TEXTS),
    ("dense (BGE-small)", _variant("dense_bge_small", "dense"), 3, TEXTS),
    ("dense (fine-tuned MiniLM)", _variant("dense_minilm_finetuned", "dense"), 3, TEXTS),
    ("best system nDCG@10", _variant("dense_minilm_finetuned__fusion_weighted", "fusion"), 3, TEXTS),
    ("best system Recall@10",
     _variant("dense_minilm_finetuned__fusion_weighted", "fusion", "Recall@10"), 3, TEXTS),
]

_quality = _json("answer_quality.json")["per_reader"]
ANSWERS = [
    (f"{reader} {criterion}", _quality[reader][criterion], 2, {"paper/main.tex": TEXTS["paper/main.tex"]})
    for reader in ("generative", "extractive")
    for criterion in ("correctness", "completeness", "faithfulness", "citation_accuracy")
]


@pytest.mark.parametrize("what, value, decimals, texts", RETRIEVAL + ANSWERS,
                         ids=[c[0] for c in RETRIEVAL + ANSWERS])
def test_number_appears_in_text(what, value, decimals, texts):
    shown = f"{value:.{decimals}f}"
    for name, text in texts.items():
        assert shown in text, f"{name} does not mention {what} = {shown} (from results/)"


def test_failure_counts_in_paper():
    counts = _json("failure_summary.json")["counts"]
    total = sum(counts.values())
    for category in ("missing_evidence", "terminology_mismatch", "reranker_error", "ambiguous_query"):
        n = counts[category]
        row = f"{n:>2} ({round(100 * n / total)}\\%)".strip()
        assert row in TEXTS["paper/main.tex"], f"failure table should show {category} as '{row}'"
