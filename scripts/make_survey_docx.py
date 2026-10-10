#!/usr/bin/env python
"""Build paper/Literature_Survey.docx from docs/LITERATURE_SURVEY.md.

    python scripts/make_survey_docx.py

The Markdown file is the source: edit it, then run this. Handles the subset the survey
uses: # and ## headings, paragraphs, a > note, numbered lists, | tables |, **bold**,
*italic* and `code`, and the numbered reference list.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "LITERATURE_SURVEY.md"
TARGET = REPO / "paper" / "Literature_Survey.docx"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_inline(paragraph, text: str, size: float | None = None) -> None:
    """Add text with **bold**, *italic* and `code` spans as separate runs."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Courier New"
        else:
            run = paragraph.add_run(part)
        if size:
            run.font.size = Pt(size)


def blocks(markdown: str):
    """Group lines into (kind, lines) blocks: heading, para, note, list, table, refs."""
    kind, lines = None, []
    for line in markdown.splitlines():
        if line.startswith("#"):
            if lines:
                yield kind, lines
            yield "heading", [line]
            kind, lines = None, []
            continue
        if not line.strip():
            if lines:
                yield kind, lines
            kind, lines = None, []
            continue
        this = ("table" if line.startswith("|") else "note" if line.startswith(">")
                else "refs" if re.match(r"^\[\d+\]", line) else "list" if re.match(r"^\s*\d+\. ", line)
                else "para")
        if kind == "list" and this == "para":  # continuation line of a list item
            this = "list"
        if kind and this != kind:
            yield kind, lines
            lines = []
        kind = this
        lines.append(line)
    if lines:
        yield kind, lines


def build():
    """The survey as a Word document, built from the Markdown source."""
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    first_heading = True
    for kind, lines in blocks(SOURCE.read_text(encoding="utf-8")):
        if kind == "heading":
            level = len(lines[0]) - len(lines[0].lstrip("#"))
            text = lines[0].lstrip("#").strip()
            if first_heading:
                title = doc.add_paragraph()
                title.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = title.add_run("LITERATURE SURVEY\n" + text.split(": ", 1)[-1])
                run.bold = True
                run.font.size = Pt(15)
                first_heading = False
            else:
                doc.add_heading(text, level=min(level - 1, 3))
        elif kind == "para":
            text = " ".join(l.strip() for l in lines)
            p = doc.add_paragraph()
            if text.startswith("Vivansh Garg") or text.startswith("Deep Learning course"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if text.startswith("**Table"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, text)
        elif kind == "note":
            p = doc.add_paragraph()
            add_inline(p, " ".join(l.lstrip("> ").strip() for l in lines), size=10)
            for run in p.runs:
                run.italic = True
        elif kind == "list":
            items, current = [], []
            for l in lines:
                if re.match(r"^\s*\d+\. ", l):
                    if current:
                        items.append(" ".join(current))
                    current = [re.sub(r"^\s*\d+\. ", "", l).strip()]
                else:
                    current.append(l.strip())
            items.append(" ".join(current))
            for item in items:
                add_inline(doc.add_paragraph(style="List Number"), item)
        elif kind == "table":
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines
                    if not re.match(r"^\|[\s\-|]+\|$", l.strip())]
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    para = table.cell(r, c).paragraphs[0]
                    add_inline(para, f"**{cell}**" if r == 0 else cell, size=8.5)
        elif kind == "refs":
            for l in lines:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Pt(24)
                p.paragraph_format.first_line_indent = Pt(-24)
                add_inline(p, l, size=10)

    return doc


def main() -> None:
    build().save(TARGET)
    print(f"written to {TARGET.relative_to(REPO)}")


if __name__ == "__main__":
    main()
