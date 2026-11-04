"""scripts/rewrite_progress.py lists every chapter of the full paper once, in the guide's order."""

import argparse
import importlib.util
import os
import re
from pathlib import Path

import pytest

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


def test_ai_wording_counts_five_word_phrases_from_the_draft():
    # how much of the AI draft's wording is still there, as a share of five-word phrases
    draft = "one two three four five six seven eight"
    assert rp.ai_wording(draft, draft) == 1.0
    assert rp.ai_wording("alpha beta gamma delta epsilon zeta", draft) == 0.0
    # 4 phrases, of which one..five and two..six are the draft's; case and comments do not count
    assert rp.ai_wording("One two three four five six % a comment\nnine ten", draft) == 0.5
    assert rp.ai_wording("one two three four", draft) is None  # fewer than five words


def test_sections_cut_at_section_and_subsection_only():
    # a long chapter is rewritten a section at a time, so it is measured that way too
    tex = ("% Drafted with AI assistance\nOpening words before any heading.\n"
           "\\section{Method}\\label{sec:m}\nThe method in short.\n"
           "\\subsection{First part}\nFirst part words.\n"
           "\\paragraph{A paragraph.} Still the first part.\n"
           "% \\subsection{Commented out}\nAnd more.\n"
           "\\section*{Empty}\\label{sec:e}\n"
           "\\subsection*{Second part}\nSecond part words.\n")
    parts = rp.sections(tex)
    assert [p["title"] for p in parts] == ["(no heading)", "Method", "First part", "Second part"]
    assert [p["words"] for p in parts] == [5, 4, 9, 3]  # "Empty" has no words, so it is left out
    # a command inside a heading does not cut the title short
    assert rp.sections("\\subsection{Stage 3: \\texttt{wRRF} fusion}\nWords.")[0]["title"] == \
        "Stage 3: \\texttt{wRRF} fusion"


def test_chapter_from_name_number_or_fragment():
    # the chapter argument need not be typed in full
    assert rp.find_chapter("03_methodology") == "03_methodology"
    assert rp.find_chapter("01") == rp.find_chapter("1") == "01_introduction"
    assert rp.find_chapter("methodology") == "03_methodology"
    assert rp.find_chapter("paper_full\\sections\\05_results.tex") == "05_results"  # tab-completed on Windows
    for arg in ("re", "12", "appendix"):  # ambiguous, no such number, no such name
        with pytest.raises(argparse.ArgumentTypeError, match="the chapters are 00_abstract"):
            rp.find_chapter(arg)


def test_chapters_are_compared_with_the_ai_draft():
    # every chapter is in the draft commit; the abstract, rewritten in #80, no longer matches it
    # in full, and a chapter still identical to its draft is 100% AI wording, section by section
    if not rp.have_draft_commit():
        if os.environ.get("GITHUB_ACTIONS"):  # tests.yml fetches the whole history, so this is a bug there
            pytest.fail(f"the AI draft (commit {rp.DRAFT_COMMIT[:7]}) is missing: is fetch-depth still 0?")
        pytest.skip(f"the AI draft (commit {rp.DRAFT_COMMIT[:7]}) is not in this clone; "
                    "git fetch --unshallow adds it")
    drafts = {name: rp.draft_text(name) for name in rp.ORDER}
    assert all(drafts.values()), [name for name, d in drafts.items() if not d]
    abstract = (REPO / "paper_full" / "sections" / "00_abstract.tex").read_text(encoding="utf-8")
    assert abstract != drafts["00_abstract"] and rp.ai_wording(abstract, drafts["00_abstract"]) < 1.0
    for name, draft in drafts.items():
        tex = (REPO / "paper_full" / "sections" / f"{name}.tex").read_text(encoding="utf-8")
        # a chapter unchanged since one of its AI drafts (for the author's abstract, the first only)
        ai_versions = [rp.text_at(rp.DRAFT_COMMIT, name)]
        if name not in rp.AUTHOR_WRITTEN:
            ai_versions.append(rp.text_at(rp.REVISION_COMMIT, name))
        if tex in ai_versions:
            assert rp.ai_wording(tex, draft) == 1.0, name
            for part in rp.sections(tex):
                assert rp.ai_wording(part["text"], draft) == 1.0, (name, part["title"])


def test_the_revision_counts_as_ai_drafted_except_the_abstract():
    # the 4 Nov revision rewrote chapters 1-8 with AI assistance: it must not read as the author's
    # own rewriting, while the author's abstract is still measured against the first draft only
    if not rp.have_draft_commit() or rp.text_at(rp.REVISION_COMMIT, "01_introduction") is None:
        pytest.skip("the AI drafts are not in this clone; git fetch --unshallow adds them")
    for name in rp.ORDER:
        revised = rp.text_at(rp.REVISION_COMMIT, name)
        assert revised is not None, name
        share = rp.ai_wording(revised, rp.draft_text(name))
        if name in rp.AUTHOR_WRITTEN:
            assert share < 1.0, name
        else:
            assert share == 1.0, name


def test_percent_is_rounded_down():
    # 100% only for unchanged text, and a shown 50% is really at least half
    assert [rp.percent(s) for s in (1.0, 0.809, 0.57, 0.4999, 0.0, None)] == ["100%", "80%", "57%", "49%", "0%", "-"]


def _table_rows(out):
    # the rows between the column header and the "AI wording:" legend, as (heading, share)
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("#  heading")) + 1
    end = next(i for i, line in enumerate(lines) if line.startswith("AI wording:")) - 1
    return [(line[4:].rsplit(None, 2)[0].strip(), line.split()[-1]) for line in lines[start:end]]


def test_one_chapter_lists_its_subsections_in_order(capsys):
    # the per-chapter view has one row per subsection, in order, in plain ASCII for Command Prompt
    tex = (REPO / "paper_full" / "sections" / "01_introduction.tex").read_text(encoding="utf-8")
    headings = re.findall(r"^\\subsection\{([^}]*)\}", tex, flags=re.M)
    assert headings
    assert rp.main(["01_introduction"]) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    assert [heading for heading, _ in _table_rows(out)] == headings


def test_one_chapter_shows_each_share_and_the_next_section(tmp_path, monkeypatch, capsys):
    # a chapter whose first section is rewritten: 0% there, 100% in the two still drafted, and
    # the next one to rewrite is the first of those
    drafted = {"Second": "the second part was written by the assistant and is still as drafted",
               "Third": "the third part was also written by the assistant and is unchanged now"}
    draft = "".join(f"\\subsection{{{t}}}\n{body}\n" for t, body in
                    [("First", "the first part as the assistant once drafted it word for word"), *drafted.items()])
    mine = "\\subsection{First}\nMy own account of this part, in different words entirely.\n" + \
        "".join(f"\\subsection{{{t}}}\n{body}\n" for t, body in drafted.items())
    (tmp_path / "09_test.tex").write_text(mine, encoding="utf-8")
    monkeypatch.setattr(rp, "SECTIONS", tmp_path)
    monkeypatch.setattr(rp, "draft_text", lambda name: draft)
    rp.show_sections("09_test")
    out = capsys.readouterr().out
    assert _table_rows(out) == [("First", "0%"), ("Second", "100%"), ("Third", "100%")]
    assert out.rstrip().splitlines()[-1] == "Next: 2  Second"
