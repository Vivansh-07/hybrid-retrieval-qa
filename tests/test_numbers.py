"""The headline numbers in the paper, README, RESULTS.md, the slides' speaker notes and the
problem statement match the result files they come from.

Every number is read from results/ and rounded the way the text reports it, so an edit
that changes a number in the text (or a rerun that changes a result) fails here until
the two agree again.
"""

import csv
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


def test_failure_selection_described_in_paper():
    """The paper says how many of the analysed claims have no relevant abstract in the top ten."""
    with open(RESULTS / "failure_analysis.csv", newline="", encoding="utf-8-sig") as fh:
        ranks = [row["gold_rank_fusion_rerank"] for row in csv.DictReader(fh)]
    # the rank of the best-ranked gold abstract; ">N" means none was retrieved
    none_in_top10 = sum(r.startswith(">") or int(r) > 10 for r in ranks)
    assert f"for {none_in_top10}, no relevant abstract in its top ten" in TEXTS["paper/main.tex"]


# The main ablation and the best system, as the other documents quote them: the slides'
# speaker notes and the problem statement (its generator's source) to three decimals,
# RESULTS.md to four, as the scripts print them.
QUOTED_IN = {
    "docs/SLIDES.md": 3,
    "scripts/make_problem_statement.py": 3,
    "RESULTS.md": 4,
}
MAIN = [
    ("BM25", _variant("", "bm25")),
    ("dense (pretrained MiniLM)", _variant("", "dense")),
    ("RRF fusion", _variant("", "fusion")),
    ("fusion + rerank", _variant("", "fusion+rerank")),
    ("dense (fine-tuned MiniLM)", _variant("dense_minilm_finetuned", "dense")),
    ("best system nDCG@10", _variant("dense_minilm_finetuned__fusion_weighted", "fusion")),
]


@pytest.mark.parametrize("what, value", MAIN, ids=[m[0] for m in MAIN])
def test_main_numbers_in_slides_problem_statement_and_results_page(what, value):
    for name, decimals in QUOTED_IN.items():
        shown = f"{value:.{decimals}f}"
        assert shown in (REPO / name).read_text(encoding="utf-8"), (
            f"{name} does not mention {what} = {shown} (from results/)")
