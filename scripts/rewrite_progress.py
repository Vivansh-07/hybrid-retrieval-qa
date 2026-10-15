#!/usr/bin/env python
"""Which chapters of the full paper are still the AI-assisted draft?

    python scripts/rewrite_progress.py

Every chapter in paper_full/sections/ starts with the comment line
"% Drafted with AI assistance"; delete it when the chapter is rewritten in your own words
(D:\\Presentation\\Rewrite_guide.md). This lists each chapter with its word count, whether the
marker is still there, and the date of its last commit, in the order the guide suggests.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SECTIONS = REPO / "paper_full" / "sections"
MARKER = "% Drafted with AI assistance"
ORDER = ["00_abstract", "01_introduction", "08_limitations_conclusion", "07_discussion",
         "02_literature_review", "06_failure_analysis", "05_results", "04_experimental_setup",
         "03_methodology"]


def words(tex: str) -> int:
    """Rough word count of the prose: comments, commands, maths and braces removed."""
    text = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\begin\{(equation|align|algorithmic|tabularx?)\}.*?\\end\{\1\}", " ", text, flags=re.S)
    text = re.sub(r"\\[A-Za-z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))


def last_commit(path: Path) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
                         cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "-"


def progress() -> list:
    rows = []
    for name in ORDER:
        path = SECTIONS / f"{name}.tex"
        tex = path.read_text(encoding="utf-8")
        rows.append({"chapter": name, "words": words(tex), "draft": MARKER in tex, "changed": last_commit(path)})
    return rows


def main() -> int:
    rows = progress()
    print(f"{'#':>2}  {'chapter':<27}{'words':>6}  {'status':<14}last commit")
    for i, r in enumerate(rows, 1):
        status = "AI draft" if r["draft"] else "rewritten"
        print(f"{i:>2}  {r['chapter']:<27}{r['words']:>6}  {status:<14}{r['changed']}")
    done = sum(not r["draft"] for r in rows)
    print(f"\n{done} of {len(rows)} chapters rewritten "
          f"({sum(r['words'] for r in rows if not r['draft'])} of {sum(r['words'] for r in rows)} words).")
    todo = [r["chapter"] for r in rows if r["draft"]]
    if todo:
        print(f"Next: {todo[0]}  (in Overleaf: \\newcommand{{\\onlychapter}}{{{todo[0]}}})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
