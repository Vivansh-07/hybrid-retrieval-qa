"""The full-length paper (paper_full/) stays compilable: every chapter, label, citation and
figure resolves. No LaTeX installation is needed; these are the mistakes that most often
break a build after an edit in Overleaf.
"""

import re
from pathlib import Path

FULL = Path(__file__).resolve().parents[1] / "paper_full"
CITE = re.compile(r"\\(?:[Pp]arencite|[Tt]extcite|cite|citeauthor|citeyear|nocite)\*?(?:\[[^\]]*\])*\{([^}]+)\}")


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _source() -> str:
    """main.tex with every \\input{...} chapter inlined, comments removed."""
    main = _strip_comments((FULL / "main.tex").read_text(encoding="utf-8"))
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda m: _strip_comments((FULL / f"{m.group(1)}.tex").read_text(encoding="utf-8")),
                  main)


def test_every_chapter_file_exists():
    main = _strip_comments((FULL / "main.tex").read_text(encoding="utf-8"))
    missing = [name for name in re.findall(r"\\input\{([^}]+)\}", main) if not (FULL / f"{name}.tex").exists()]
    assert not missing, f"\\input files missing: {missing}"


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
