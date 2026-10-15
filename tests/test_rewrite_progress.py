"""scripts/rewrite_progress.py lists every chapter of the full paper once, in the guide's order."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rewrite_progress", REPO / "scripts" / "rewrite_progress.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def test_every_chapter_is_listed_once():
    files = sorted(p.stem for p in (REPO / "paper_full" / "sections").glob("*.tex"))
    assert sorted(rp.ORDER) == files


def test_word_count_ignores_latex():
    tex = ("% Drafted with AI assistance\n\\section{Intro}\\label{sec:x}\n"
           "Fusion helps, as \\textcite{a} showed: $s = 1/(k + r)$ for rank $r$.\n"
           "\\begin{equation}x = y\\end{equation}\n")
    assert rp.words(tex) == 6  # Fusion helps as showed for rank


def test_progress_reports_the_marker():
    rows = rp.progress()
    assert len(rows) == len(rp.ORDER)
    for r in rows:
        text = (REPO / "paper_full" / "sections" / f"{r['chapter']}.tex").read_text(encoding="utf-8")
        assert r["draft"] == (rp.MARKER in text)
        assert r["words"] > 0
