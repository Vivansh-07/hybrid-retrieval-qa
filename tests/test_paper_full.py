"""The full-length paper (paper_full/) stays compilable: every chapter, label, citation and
figure resolves. No LaTeX installation is needed; these are the mistakes that most often
break a build after an edit in Overleaf.
"""

import re
from pathlib import Path

FULL = Path(__file__).resolve().parents[1] / "paper_full"
CITE = re.compile(r"\\(?:[Pp]arencite|[Tt]extcite|cite|citeauthor|citeyear|nocite)\*?(?:\[[^\]]*\])*\{([^}]+)\}")
# chapters are included as \chapterfile{name} (sections/name.tex), other files as \input{path}
INCLUDE = re.compile(r"\\chapterfile\{([^}]+)\}|\\input\{([^}#]+)\}")

CHAPTERS = ["00_abstract", "01_introduction", "02_literature_review", "03_methodology",
            "04_experimental_setup", "05_results", "06_failure_analysis", "07_discussion",
            "08_limitations_conclusion"]


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _main() -> str:
    return _strip_comments((FULL / "main.tex").read_text(encoding="utf-8"))


def _included(main: str) -> list:
    return [f"sections/{chapter}" if chapter else path for chapter, path in INCLUDE.findall(main)]


def _source() -> str:
    """main.tex with every included chapter inlined, comments removed."""
    return INCLUDE.sub(
        lambda m: _strip_comments((FULL / f"{'sections/' + m.group(1) if m.group(1) else m.group(2)}.tex")
                                  .read_text(encoding="utf-8")),
        _main())


def test_every_chapter_file_exists():
    missing = [name for name in _included(_main()) if not (FULL / f"{name}.tex").exists()]
    assert not missing, f"chapter files missing: {missing}"


def test_every_chapter_is_included_in_order():
    assert _included(_main()) == [f"sections/{c}" for c in CHAPTERS]


def test_the_whole_paper_is_compiled():
    """\\onlychapter compiles a single chapter in Overleaf; the saved version must build them all."""
    match = re.search(r"\\newcommand\{\\onlychapter\}\{([^}]*)\}", _main())
    assert match, "main.tex should define \\onlychapter"
    assert match.group(1).strip() == "", (
        f"main.tex compiles only '{match.group(1)}': set it back to \\newcommand{{\\onlychapter}}{{}}")


def test_every_ref_has_a_label():
    tex = _source()
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", tex))
    assert refs <= labels, f"\\ref without \\label: {sorted(refs - labels)}"


def test_every_citation_is_in_the_bibliography():
    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", (FULL / "references.bib").read_text(encoding="utf-8")))
    cited = {k.strip() for group in CITE.findall(_source()) for k in group.split(",")}
    assert cited <= keys, f"citation keys missing from references.bib: {sorted(cited - keys)}"


def test_every_figure_file_exists():
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", _source()):
        assert (FULL.parent / "paper" / "figures" / name).exists() or (FULL / name).exists(), \
            f"missing figure: {name} (looked in paper/figures and paper_full)"


def test_environments_and_braces_balance():
    tex = _source()
    begins = re.findall(r"\\begin\{([^}]+)\}", tex)
    ends = re.findall(r"\\end\{([^}]+)\}", tex)
    assert sorted(begins) == sorted(ends), "unbalanced \\begin/\\end"
    text = re.sub(r"\\[{}]", "", tex)
    assert text.count("{") == text.count("}"), "unbalanced braces"
