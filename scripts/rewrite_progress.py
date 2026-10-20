#!/usr/bin/env python
"""Which chapters of the full paper are still the AI-assisted draft?

    python scripts/rewrite_progress.py                    every chapter
    python scripts/rewrite_progress.py 01_introduction    one chapter, section by section
                                                          (also 01, 1 or introduction)

Every chapter in paper_full/sections/ starts with the comment line
"% Drafted with AI assistance"; delete it when the chapter is rewritten in your own words
(D:\\Presentation\\Rewrite_guide.md). This lists each chapter with its word count, whether the
marker is still there, the date of its last commit and its AI wording, in the order the guide
suggests. Given a chapter, it lists that chapter's sections and subsections instead, so that a
long chapter can be rewritten and pushed a section at a time, and names the first section that
is still at least half AI wording.

AI wording is the share of the text's five-word phrases that also occur in the chapter's AI
draft (commit 16ed095, the last one before the first rewrite). 100% is the draft unchanged, and
a light edit stays high. A low share shows that the draft's wording is gone; it does not by
itself prove the words are your own, so only the marker decides "rewritten".
"""
import argparse
import math
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
# The last commit on main before the first rewrite (#80, 15 Oct): every chapter there is the
# complete AI draft. A fixed commit, so a chapter edited twice is still compared with the draft.
DRAFT_COMMIT = "16ed095c5409fb3f2f163cb27a1445ac7985825d"
N = 5               # words per phrase: long enough that a shared phrase is the draft's, not common English
STILL_DRAFT = 0.5   # a section with at least this share of AI wording is the next one to rewrite
NO_HEADING = "(no heading)"
HEADING = re.compile(r"\\(?:section|subsection)\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}")  # \texttt{..} inside too
HOW_TO_READ = ("AI wording: share of five-word phrases from the AI draft; 100% = unchanged, "
               "a light edit stays high.\n"
               "A low share shows the draft's wording is gone; it does not by itself prove the words "
               "are your own.")


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def prose(tex: str) -> list:
    """The prose words in order: comments, commands, maths and braces removed."""
    text = re.sub(r"\$[^$]*\$", " ", strip_comments(tex))
    text = re.sub(r"\\begin\{(equation|align|algorithmic|tabularx?)\}.*?\\end\{\1\}", " ", text, flags=re.S)
    text = re.sub(r"\\[A-Za-z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def words(tex: str) -> int:
    """Rough word count of the prose: comments, commands, maths and braces removed."""
    return len(prose(tex))


def phrases(tex: str) -> list:
    """Every run of N consecutive prose words, lowercased, overlapping."""
    w = [x.lower() for x in prose(tex)]
    return [tuple(w[i:i + N]) for i in range(len(w) - N + 1)]


def ai_wording(tex: str, draft_tex: str):
    """Share of the text's five-word phrases that also occur in the draft; None below five words.

    The text's phrases are counted with repetition, the draft's as a set, so draft wording that
    was only moved, within the chapter or to another section, still counts.
    """
    mine = phrases(tex)
    if not mine:
        return None
    theirs = set(phrases(draft_tex))
    return sum(p in theirs for p in mine) / len(mine)


def sections(tex: str) -> list:
    """The chapter cut at each \\section and \\subsection (not \\paragraph); parts with no prose left out.

    Text before the first heading is its own part, and so is a section's text before its first
    subsection. Commented-out headings do not cut, because the comments are removed first.
    """
    text = strip_comments(tex)
    cuts = list(HEADING.finditer(text))
    starts = [0] + [m.start() for m in cuts] + [len(text)]
    titles = [NO_HEADING] + [" ".join(m.group(1).split()) for m in cuts]
    parts = []
    for title, start, end in zip(titles, starts, starts[1:]):
        n = words(text[start:end])
        if n:
            parts.append({"title": title, "words": n, "text": text[start:end]})
    return parts


def find_chapter(arg: str) -> str:
    """A chapter from its full name, its number (01 or 1) or a unique part of its name."""
    key = re.sub(r"\.tex$", "", re.split(r"[\\/]", arg)[-1]).lower()   # also a tab-completed path
    if key.isdigit():
        found = [c for c in ORDER if int(c[:2]) == int(key)]
    else:
        found = [c for c in ORDER if c == key] or [c for c in ORDER if key in c]
    if len(found) != 1:
        why = "matches no chapter" if not found else "matches " + ", ".join(sorted(found))
        raise argparse.ArgumentTypeError(f"'{arg}' {why}; the chapters are " + ", ".join(sorted(ORDER)))
    return found[0]


def last_commit(path: Path) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
                         cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "-"


def draft_text(name: str):
    """The chapter as the AI drafted it, or None if this clone lacks the draft commit or the file."""
    out = subprocess.run(["git", "show", f"{DRAFT_COMMIT}:paper_full/sections/{name}.tex"],
                         cwd=REPO, capture_output=True, encoding="utf-8", errors="replace")
    return out.stdout if out.returncode == 0 else None


def have_draft_commit() -> bool:
    out = subprocess.run(["git", "cat-file", "-e", f"{DRAFT_COMMIT}^{{commit}}"], cwd=REPO, capture_output=True)
    return out.returncode == 0


def missing_draft_note(names: list) -> str:
    """Why AI wording shows "-": a shallow clone or ZIP download, or a chapter newer than the draft."""
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO, capture_output=True)
    if inside.returncode != 0:
        return "Note: this folder is not a git clone; clone the repository to compare with the AI draft."
    if not have_draft_commit():
        return (f"Note: the AI draft (commit {DRAFT_COMMIT[:7]}) is not in this clone; "
                "git fetch --unshallow adds it.")
    return f"Note: {', '.join(names)} did not exist in the AI draft (commit {DRAFT_COMMIT[:7]})."


def percent(share) -> str:
    """Rounded down, so that 100% means unchanged and 50% means at least half."""
    return "-" if share is None else f"{math.floor(share * 100 + 1e-9)}%"   # 1e-9: 0.57 * 100 is 56.999...


def progress() -> list:
    rows = []
    for name in ORDER:
        path = SECTIONS / f"{name}.tex"
        tex = path.read_text(encoding="utf-8")
        old = draft_text(name)
        rows.append({"chapter": name, "words": words(tex), "draft": MARKER in tex, "changed": last_commit(path),
                     "ai": None if old is None else ai_wording(tex, old), "in_draft": old is not None})
    return rows


def show_chapters() -> None:
    rows = progress()
    print(f"{'#':>2}  {'chapter':<27}{'words':>6}  {'status':<14}{'last commit':<13}AI wording")
    for i, r in enumerate(rows, 1):
        status = "AI draft" if r["draft"] else "rewritten"
        print(f"{i:>2}  {r['chapter']:<27}{r['words']:>6}  {status:<14}{r['changed']:<13}{percent(r['ai']):>10}")
    print("\n" + HOW_TO_READ)
    missing = [r["chapter"] for r in rows if not r["in_draft"]]
    if missing:
        print(missing_draft_note(missing))
    done = sum(not r["draft"] for r in rows)
    print(f"\n{done} of {len(rows)} chapters rewritten "
          f"({sum(r['words'] for r in rows if not r['draft'])} of {sum(r['words'] for r in rows)} words).")
    todo = [r["chapter"] for r in rows if r["draft"]]
    if todo:
        print(f"Next: {todo[0]}  (in Overleaf: \\newcommand{{\\onlychapter}}{{{todo[0]}}})")


def show_sections(name: str) -> None:
    tex = (SECTIONS / f"{name}.tex").read_text(encoding="utf-8")
    old = draft_text(name)
    parts = sections(tex)
    for p in parts:   # against the whole draft chapter, so text moved to another section still counts
        p["ai"] = None if old is None else ai_wording(p["text"], old)
        p["shown"] = p["title"].encode("ascii", "replace").decode()   # plain ASCII for Command Prompt
    width = max([len("heading")] + [len(p["shown"]) for p in parts]) + 2
    print(f"{name}: {words(tex)} words, AI wording {percent(None if old is None else ai_wording(tex, old))}\n")
    print(f"{'#':>2}  {'heading':<{width}}{'words':>6}  AI wording")
    for i, p in enumerate(parts, 1):
        print(f"{i:>2}  {p['shown']:<{width}}{p['words']:>6}  {percent(p['ai']):>10}")
    print("\n" + HOW_TO_READ)
    if old is None:
        print(missing_draft_note([name]))
        return
    todo = [(i, p) for i, p in enumerate(parts, 1) if p["ai"] is not None and p["ai"] >= STILL_DRAFT]
    if todo:
        print(f"\nNext: {todo[0][0]}  {todo[0][1]['shown']}")
    else:
        print(f"\nNo section is left at or above {STILL_DRAFT:.0%} AI wording.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapter", nargs="?", type=find_chapter,
                    help="one chapter, section by section: its name, number or a part of its name")
    args = ap.parse_args(argv)
    if args.chapter:
        show_sections(args.chapter)
    else:
        show_chapters()
    return 0


if __name__ == "__main__":
    sys.exit(main())
