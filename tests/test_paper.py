"""The paper's LaTeX source stays compilable: every reference, citation and figure resolves.

No LaTeX installation is needed; these are the mistakes that most often break an
Overleaf build after an edit.
"""

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def _source() -> str:
    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    # drop comments (an unescaped % to the end of the line)
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def test_every_ref_has_a_label():
    tex = _source()
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    assert refs <= labels, f"\\ref without \\label: {sorted(refs - labels)}"


def test_every_citation_is_in_the_bibliography():
    tex = _source()
    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", (PAPER / "references.bib").read_text(encoding="utf-8")))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for k in group.split(",")}
    assert cited <= keys, f"\\cite keys missing from references.bib: {sorted(cited - keys)}"


def test_every_figure_file_exists():
    tex = _source()
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        assert (PAPER / name).exists(), f"missing figure file: paper/{name}"


def test_environments_and_braces_balance():
    tex = _source()
    for env in set(re.findall(r"\\begin\{([^}]+)\}", tex)):
        opened = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", tex))
        closed = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", tex))
        assert opened == closed, f"\\begin{{{env}}} x{opened} but \\end x{closed}"
    unescaped = re.sub(r"\\[{}]", "", tex)
    assert unescaped.count("{") == unescaped.count("}"), "unbalanced braces"
