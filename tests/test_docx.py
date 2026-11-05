"""scripts/make_docx.py flattens the full paper into plain LaTeX for pandoc's Word writer.

These tests need no pandoc: they check the flattening, which is where a change to the paper can
silently break the Word version (an unresolved reference, a lost algorithm line, a wrong number).
"""

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_docx", REPO / "scripts" / "make_docx.py")
make_docx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_docx)

DOC, META = make_docx.flattened()


def test_every_reference_is_resolved():
    assert not re.search(r"\\(eq)?ref\{|\\label\{|\\chapterfile|\\paragraph\{", DOC)
    assert "\\begin{algorithm" not in DOC and "tikzpicture" not in DOC
    assert "\\begin{tabularx}" not in DOC and "\\begin{description}" not in DOC


def test_numbers_follow_latex_order():
    numbers = make_docx.number_labels("\n".join(
        make_docx.strip_comments((REPO / "paper_full" / "sections" / f"{c}.tex").read_text(encoding="utf-8"))
        for c in make_docx.CHAPTERS))
    assert numbers["sec:introduction"] == "1" and numbers["sec:problem"] == "1.2"
    assert numbers["tab:notation"] == "1" and numbers["fig:pipeline"] == "1"
    assert numbers["alg:bm25"] == "1" and numbers["eq:npassages"] == "1"
    # the two-row align numbers each row, as LaTeX does
    assert int(numbers["eq:dcg"]) == int(numbers["eq:recall"]) + 1
    # every float and equation gets a distinct number within its kind
    for prefix in ("tab:", "fig:", "alg:", "eq:"):
        values = [v for k, v in numbers.items() if k.startswith(prefix)]
        assert len(values) == len(set(values)), prefix


def test_algorithms_keep_every_step():
    # Algorithm 3 (fine-tuning) has 10 numbered steps in the PDF
    block = DOC.split("\\textbf{Algorithm 3.}", 1)[1].split("\n\n", 1)[0]
    assert [int(n) for n in re.findall(r"\{(\d+):\}", block)] == list(range(1, 11))
    assert "\\textbf{for}" in block and "\\textbf{return}" in block


def test_front_matter_and_declarations():
    assert "Pooja Gupta, Vivansh Garg" in META and "pooja.gupta@jaipur.manipal.edu" in META
    assert "\\section*{Abstract}" in DOC and "\\textbf{Keywords:}" in DOC
    assert "\\section*{Declarations}" in DOC and "Use of generative AI" in DOC
