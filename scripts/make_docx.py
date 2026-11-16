#!/usr/bin/env python
"""Builds a Word version of the full paper (paper_full/) with pandoc.

    python scripts/make_docx.py [output.docx]

pandoc (pip install pypandoc_binary, inside the project environment) reads LaTeX, but not this
paper's own macros, its algorithm environments or the TikZ figure. So the paper is first
flattened into plain LaTeX that pandoc converts faithfully:
  - section, table, figure, algorithm and equation numbers are resolved here, in the order LaTeX
    assigns them, and every \\ref and \\eqref is replaced by its number;
  - every display equation becomes a native Word equation with its number after it;
  - each algorithm becomes a numbered list of pseudocode lines, indented like the PDF;
  - Figure 1 (TikZ) is replaced by paper_full/word/pipeline_figure.png, cut from the PDF;
  - citations stay biblatex citations, which pandoc formats in APA 7 (paper_full/word/apa.csl,
    the Citation Style Language's APA 7th edition style) from paper_full/references.bib.
The title, authors, affiliation, e-mail addresses and Declarations come from paper_full/main.tex.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FULL = REPO / "paper_full"
WORD = FULL / "word"
OUT = REPO / "data" / "tmp" / "word" / "paper.docx"
CHAPTERS = ["01_introduction", "02_literature_review", "03_methodology", "04_experimental_setup",
            "05_results", "06_failure_analysis", "07_discussion", "08_limitations_conclusion"]
NBSP = "~"


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def braced(text: str, start: int):
    """The content of the {...} group opening at text[start], and the index after it."""
    assert text[start] == "{", text[start:start + 30]
    depth, i = 0, start
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
        if depth == 0:
            return text[start + 1:i - 1], i


# --- numbering --------------------------------------------------------------------------------

EVENT = re.compile(r"\\(section|subsection)(\*?)\{|\\begin\{(table|figure|algorithm|equation|multline|align)\}"
                   r"|\\end\{(align)\}|\\label\{([^}]+)\}|(\\\\)")


def number_labels(body: str) -> dict:
    """label -> its number as LaTeX prints it (sections 1, 1.2; floats and equations in source order)."""
    numbers, sec, sub = {}, 0, 0
    count = {"table": 0, "figure": 0, "algorithm": 0, "equation": 0}
    current, in_align = None, False   # what the next \label refers to
    for m in EVENT.finditer(body):
        kind, star, env, end_env, label, newline = m.groups()
        if kind:
            if star:
                current = None
                continue
            if kind == "section":
                sec, sub = sec + 1, 0
                current = str(sec)
            else:
                sub += 1
                current = f"{sec}.{sub}"
        elif env in ("table", "figure", "algorithm"):
            count[env] += 1
            current = str(count[env])
        elif env in ("equation", "multline"):
            count["equation"] += 1
            current = str(count["equation"])
        elif env == "align":
            in_align = True
            count["equation"] += 1
            current = str(count["equation"])
        elif end_env == "align":
            in_align = False
        elif newline and in_align:      # a new row of an align is a new equation
            count["equation"] += 1
            current = str(count["equation"])
        elif label:
            numbers[label] = current
    return numbers


# --- algorithms -------------------------------------------------------------------------------

def algorithm_lines(body: str) -> list:
    """(indent, number or None, text) for each line of an algorithmic environment."""
    lines, depth, n = [], 0, 0

    def clean(s: str) -> str:
        s = re.sub(r"\\Comment\{", r"\\quad $\\triangleright$~\\emph{", s)
        # \Call{F}{x} -> F(x); the paper uses \Call only inside $...$, so a maths operator name
        s = re.sub(r"\\Call\{([^}]*)\}\{", r"\\operatorname{\1}(", s)
        s = s.replace("\\Return", "\\textbf{return}")
        return s.strip()

    def call_args(s: str) -> str:
        # close the "(" opened for \Call: the matching "}" of its argument becomes ")"
        out, i = "", 0
        while i < len(s):
            if s.startswith("\\operatorname{", i):
                name, j = braced(s, i + len("\\operatorname"))
                if s[j:j + 1] == "(":
                    arg, k = braced("{" + s[j + 1:], 0)
                    out += f"\\operatorname{{{name}}}({arg})"
                    i = j + 1 + len(arg) + 1
                    continue
            out += s[i]
            i += 1
        return out

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"\\(Require|Ensure)\s*(.*)", line)
        if m:
            lines.append((0, None, f"\\textbf{{{'Input' if m.group(1) == 'Require' else 'Output'}:}} {clean(m.group(2))}"))
            continue
        if line.startswith("\\Statex"):
            lines.append((depth, None, clean(line[len("\\Statex"):])))
            continue
        m = re.match(r"\\(For|If|While)\{", line)
        if m:
            cond, j = braced(line, m.end() - 1)
            rest = line[j:].strip()
            word = {"For": "for", "If": "if", "While": "while"}[m.group(1)]
            then = "then" if word == "if" else "do"
            end = f"\\End{m.group(1)}"
            n += 1
            if rest.endswith(end):      # the whole block on one line
                inner = rest[:-len(end)].strip()
                lines.append((depth, n, f"\\textbf{{{word}}} {cond} \\textbf{{{then}}} {clean(inner)}"))
            else:
                lines.append((depth, n, f"\\textbf{{{word}}} {cond} \\textbf{{{then}}}"))
                depth += 1
            continue
        if re.match(r"\\End(For|If|While)", line):
            depth -= 1
            continue
        if line.startswith("\\State") or line.startswith("\\Return"):
            n += 1
            lines.append((depth, n, clean(line[len("\\State"):] if line.startswith("\\State") else line)))
            continue
        lines.append((depth, None, clean(line)))   # a continuation
    return [(d, k, call_args(t).replace("\\label{line:break}", "")) for d, k, t in lines]


def convert_algorithms(body: str, numbers: dict) -> str:
    def one(m):
        env = m.group(1)
        caption, _ = braced(env, env.index("\\caption{") + len("\\caption"))
        label = re.search(r"\\label\{([^}]+)\}", env).group(1)
        inner = re.search(r"\\begin\{algorithmic\}(?:\[\d\])?(.*?)\\end\{algorithmic\}", env, re.S).group(1)
        # one paragraph, one line per step (a number right after \noindent, or a bare "1:" at the
        # start of a line, is dropped by pandoc's reader, so the number is braced)
        rows = [f"\\textbf{{Algorithm {numbers[label]}.}} {caption}"]
        for depth, k, text in algorithm_lines(inner):
            prefix = f"{{{k}:}}" if k else ""
            rows.append(f"{prefix}{NBSP * (2 + 4 * depth)}{text}")
        return "\n\n" + "\\newline\n".join(rows) + "\n\n"
    return re.sub(r"\\begin\{algorithm\}(?:\[[^\]]*\])?(.*?)\\end\{algorithm\}", one, body, flags=re.S)


# --- equations, floats, references -------------------------------------------------------------

EQ_MARK = "EQNO{}EQNO"   # carried inside the Word equation; finish() moves it to the right margin


def convert_equations(body: str, numbers: dict) -> str:
    def tagged(math: str, label: str) -> str:
        return f"\n\\[\n{math.strip()} \\text{{{EQ_MARK.format(numbers[label])}}}\n\\]\n"

    def equation(m):
        inner = m.group(2)
        label = re.search(r"\\label\{([^}]+)\}", inner).group(1)
        inner = re.sub(r"\\label\{[^}]+\}", "", inner)
        if m.group(1) == "multline":
            inner = " ".join(part.strip() for part in inner.split("\\\\"))
        return tagged(inner, label)

    def align(m):
        out = []
        for row in m.group(1).split("\\\\"):
            label = re.search(r"\\label\{([^}]+)\}", row).group(1)
            row = re.sub(r"\\label\{[^}]+\}", "", row)
            row = row.replace("&=", "=").replace("&", "\\qquad ")
            out.append(tagged(row, label))
        return "".join(out)

    body = re.sub(r"\\begin\{(equation|multline)\}(.*?)\\end\{\1\}", equation, body, flags=re.S)
    return re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", align, body, flags=re.S)


def convert_floats(body: str, numbers: dict) -> str:
    def one(m):
        kind, env = m.group(1), m.group(2)
        label = re.search(r"\\label\{([^}]+)\}", env).group(1)
        start = env.index("\\caption{")
        caption, end = braced(env, start + len("\\caption"))
        name = "Table" if kind == "table" else "Figure"
        env = env[:start] + f"\\caption{{{name} {numbers[label]}. {caption}}}" + env[end:]
        if "tikzpicture" in env:
            env = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                         r"\\includegraphics[width=\\textwidth]{pipeline_figure.png}", env, flags=re.S)
        # tabularx and the paper's own column types (L, P{..}) -> a plain tabular
        def spec(columns: str) -> str:
            columns = columns.replace("@{}", "")
            return re.sub(r"[Pp]\{[^}]*\}|[LX]", "l", columns)
        column_spec = r"\{((?:[^{}]|\{[^{}]*\})*)\}"
        env = re.sub(r"\\begin\{tabularx\}\{[^}]*\}" + column_spec,
                     lambda t: "\\begin{tabular}{" + spec(t.group(1)) + "}", env)
        env = re.sub(r"\\begin\{tabular\}" + column_spec,
                     lambda t: "\\begin{tabular}{" + spec(t.group(1)) + "}", env)
        env = env.replace("\\end{tabularx}", "\\end{tabular}")
        env = re.sub(r"\\(small|footnotesize|centering|addlinespace)\b", "", env)
        env = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", env)
        return f"\\begin{{{kind}}}{env}\\end{{{kind}}}"
    return re.sub(r"\\begin\{(table|figure)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}", one, body, flags=re.S)


def resolve_references(body: str, numbers: dict) -> str:
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({numbers[m.group(1)]})", body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: numbers[m.group(1)], body)
    return re.sub(r"\\label\{[^}]+\}", "", body)


def research_questions(body: str) -> str:
    """The description list of research questions -> one paragraph each, "RQ1" in bold."""
    def one(m):
        items = re.findall(r"\\item\[([^\]]+)\]\s*(.*?)(?=\\item\[|$)", m.group(1), flags=re.S)
        return "\n\n".join(f"\\textbf{{{name}}}~{' '.join(text.split())}" for name, text in items) + "\n\n"
    return re.sub(r"\\begin\{description\}(.*?)\\end\{description\}", one, body, flags=re.S)


def run_in_headings(body: str) -> str:
    """\\paragraph{X} is a run-in heading in the PDF: bold text at the start of its paragraph."""
    out, i = "", 0
    for m in re.finditer(r"\\paragraph\{", body):
        title, end = braced(body, m.end() - 1)
        out += body[i:m.start()] + f"\\textbf{{{title}}}~"
        i = end
    return out + body[i:]


# --- the document ---------------------------------------------------------------------------------

def front_matter(main: str):
    title = re.search(r"\\title\{\\textbf\{(.*?)\}\}\s*\n", main, flags=re.S).group(1)
    title = " ".join(title.replace("\\\\", " ").split())
    block = re.search(r"\\author\{(.*?)\}\s*\n\\date", main, flags=re.S).group(1)
    lines = [" ".join(re.sub(r"\\normalsize|\\texttt|\\\\(\[[^]]*\])?|[{}]", " ", line).split())
             for line in block.splitlines()]
    return title, [line for line in lines if line]


def abstract_and_keywords():
    tex = strip_comments((FULL / "sections" / "00_abstract.tex").read_text(encoding="utf-8"))
    body = tex.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    text, _, keywords = body.partition("\\textbf{Keywords:}")
    text = " ".join(text.replace("\\noindent", " ").replace("\\medskip", " ").split())
    return text, " ".join(keywords.split())


def declarations(main: str) -> str:
    block = main.split("\\section*{Declarations}", 1)[1].split("\\clearpage", 1)[0]
    return re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", block)


def flattened() -> tuple:
    main = strip_comments((FULL / "main.tex").read_text(encoding="utf-8"))
    title, author_lines = front_matter(main)
    abstract, keywords = abstract_and_keywords()
    body = "\n".join(strip_comments((FULL / "sections" / f"{c}.tex").read_text(encoding="utf-8"))
                     for c in CHAPTERS)
    body += "\n\\section*{Declarations}\n" + declarations(main)
    numbers = number_labels(body)
    body = convert_algorithms(body, numbers)
    body = convert_equations(body, numbers)
    body = convert_floats(body, numbers)
    body = resolve_references(body, numbers)
    body = research_questions(body)
    body = run_in_headings(body)
    body = body.replace("\\hfill$\\square$", "$\\square$")
    head = (f"\\section*{{Abstract}}\n{abstract}\n\n\\noindent\\textbf{{Keywords:}} {keywords}\n\n")
    doc = "\\documentclass{article}\n\\begin{document}\n" + head + body + "\n\\end{document}\n"
    meta = "---\n" + f'title: "{title}"\n' + "author:\n" + "".join(f'  - "{line}"\n' for line in author_lines) \
        + "reference-section-title: References\nlink-citations: false\n...\n"
    return doc, meta


def reference_docx(pandoc: str, path: Path) -> None:
    """pandoc's default Word template, set in Times New Roman 12 pt with black headings, on A4."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor
    subprocess.run([pandoc, "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(path)
    for style in doc.styles:
        if getattr(style, "font", None) is None:
            continue
        name = style.name
        if name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract", "Author", "Title",
                    "Subtitle", "Bibliography", "Table Caption", "Image Caption", "Caption", "Date") \
                or name.startswith("Heading"):
            style.font.name = "Times New Roman"
            rpr = style.element.get_or_add_rPr()
            fonts = rpr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")
            if fonts is not None:
                for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
                    fonts.attrib.pop(f"{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}{attr}", None)
            if name.startswith("Heading") or name == "Title":
                style.font.color.rgb = RGBColor(0, 0, 0)
                style.font.bold = True
    sizes = {"Normal": 12, "Body Text": 12, "First Paragraph": 12, "Title": 18, "Author": 12,
             "Heading 1": 14, "Heading 2": 12, "Table Caption": 10, "Image Caption": 10, "Bibliography": 11}
    for name, size in sizes.items():
        if name in [s.name for s in doc.styles]:
            doc.styles[name].font.size = Pt(size)
    # paragraphs as in the PDF: no gap between them, the first after a heading flush, later ones indented
    for name in ("Body Text", "First Paragraph"):
        if name in [s.name for s in doc.styles]:
            fmt = doc.styles[name].paragraph_format
            fmt.line_spacing = 1.5
            fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            fmt.space_before, fmt.space_after = Pt(0), Pt(0)
            fmt.first_line_indent = Cm(0.75) if name == "Body Text" else Cm(0)
    if "Bibliography" in [s.name for s in doc.styles]:   # APA: hanging indent of 1.27 cm, as in the PDF
        fmt = doc.styles["Bibliography"].paragraph_format
        fmt.left_indent, fmt.first_line_indent = Cm(1.27), Cm(-1.27)
        fmt.space_after = Pt(6)
    for name, (before, after) in {"Heading 1": (18, 8), "Heading 2": (12, 6), "Heading 3": (10, 4)}.items():
        if name in [s.name for s in doc.styles]:
            fmt = doc.styles[name].paragraph_format
            fmt.space_before, fmt.space_after = Pt(before), Pt(after)
            fmt.keep_with_next = True
    for name in ("Table Caption", "Image Caption"):
        if name in [s.name for s in doc.styles]:
            doc.styles[name].font.italic = False
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin = section.right_margin = Cm(2.5)
        section.top_margin = section.bottom_margin = Cm(2.5)
    doc.save(path)


def finish(path: Path) -> None:
    """The PDF's layout where pandoc cannot express it: ruled tables and algorithms, equation numbers
    at the right margin, page numbers, a centred abstract on page 1, the Introduction and the
    References on new pages, a regular-weight author block."""
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    def border(parent, edges):
        for edge, size in edges.items():
            el = parent.find(qn(f"w:{edge}"))
            if el is None:
                el = OxmlElement(f"w:{edge}")
                parent.append(el)
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(size))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")

    doc = Document(path)
    for table in doc.tables:   # booktabs look: a thick rule above and below, a thin one under the header
        tbl_pr = table._tbl.tblPr
        borders = tbl_pr.find(qn("w:tblBorders"))
        if borders is None:
            borders = OxmlElement("w:tblBorders")
            tbl_pr.append(borders)
        border(borders, {"top": 12, "bottom": 12})
        for cell in table.rows[0].cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_borders = tc_pr.find(qn("w:tcBorders"))
            if tc_borders is None:
                tc_borders = OxmlElement("w:tcBorders")
                tc_pr.append(tc_borders)
            border(tc_borders, {"bottom": 6})
        after = table._tbl.getnext()   # air between a table and the text below it
        if after is not None and after.tag == qn("w:p"):
            spacing = after.get_or_add_pPr().find(qn("w:spacing"))
            if spacing is None:
                spacing = OxmlElement("w:spacing")
                after.get_or_add_pPr().append(spacing)
            spacing.set(qn("w:before"), "200")
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.first_line_indent = Cm(0)   # body text's indent does not belong in cells
                    for run in p.runs:
                        run.font.size = Pt(10)
    for p in doc.paragraphs:
        if p.style.name == "Author":
            for run in p.runs:
                run.font.bold = False
        if re.match(r"Algorithm \d+\.", p.text):   # a ruled box, single-spaced, like the PDF
            fmt = p.paragraph_format
            fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT   # justified lines with breaks would stretch apart
            fmt.first_line_indent = Cm(0)
            fmt.line_spacing = 1.15
            fmt.space_before, fmt.space_after = Pt(6), Pt(10)
            p_pr = p._p.get_or_add_pPr()
            p_bdr = p_pr.find(qn("w:pBdr"))
            if p_bdr is None:
                p_bdr = OxmlElement("w:pBdr")
                p_pr.append(p_bdr)
            border(p_bdr, {"top": 8, "bottom": 8})
            for run in p.runs:
                run.font.size = Pt(11)

    # a paragraph opening with a bold run-in heading ("Proof.", "RQ1") starts flush, as in the PDF
    for p in doc.paragraphs:
        if p.style.name in ("Body Text", "First Paragraph") and p.runs and p.runs[0].bold:
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(6)

    # title block: a little air between the lines, as in the PDF
    for p in doc.paragraphs:
        if p.style.name in ("Title", "Author"):
            p.paragraph_format.space_after = Pt(4 if p.style.name == "Author" else 10)

    # abstract: centred heading, indented 11-pt block, then the Introduction on a new page
    paragraphs = doc.paragraphs
    for i, p in enumerate(paragraphs):
        if p.style.name.startswith("Heading") and p.text.strip() == "Abstract":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for q in paragraphs[i + 1:]:
                if q.style.name.startswith("Heading"):
                    break
                q.paragraph_format.left_indent = q.paragraph_format.right_indent = Cm(1.0)
                q.paragraph_format.first_line_indent = Cm(0)
                q.paragraph_format.line_spacing = 1.15
                q.paragraph_format.space_after = Pt(6)
                for run in q.runs:
                    run.font.size = Pt(11)
        if p.style.name == "Heading 1" and p.text.strip().endswith(("Introduction", "References")):
            p.paragraph_format.page_break_before = True

    # equation numbers at the right margin: each numbered display equation goes into the middle
    # cell of an invisible three-cell row, with its number in the right-hand cell
    m_t = qn("m:t")
    for p in list(doc.paragraphs):
        marks = [t for t in p._p.iter(m_t) if t.text and re.fullmatch(EQ_MARK.format(r"(\d+)"), t.text)]
        if not marks:
            continue
        number = re.fullmatch(EQ_MARK.format(r"(\d+)"), marks[0].text).group(1)
        run = marks[0].getparent()
        run.getparent().remove(run)
        table = doc.add_table(rows=1, cols=3)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_pr = table._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)
        no_borders = OxmlElement("w:tblBorders")   # invisible: the row only positions the number
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "nil")
            no_borders.append(el)
        tbl_pr.append(no_borders)
        for cell, width in zip(table.rows[0].cells, (Cm(1.6), Cm(12.8), Cm(1.6))):
            cell.width = width
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            # the table style rules its first row; cell borders override the style
            tc_borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right"):
                el = OxmlElement(f"w:{edge}")
                el.set(qn("w:val"), "nil")
                tc_borders.append(el)
            cell._tc.get_or_add_tcPr().append(tc_borders)
        p._p.addnext(table._tbl)
        middle = table.rows[0].cells[1]
        middle._tc.remove(middle.paragraphs[0]._p)
        middle._tc.append(p._p)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(3)
        right = table.rows[0].cells[2].paragraphs[0]
        right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        right.add_run(f"({number})")
        # the text after a display equation continues the sentence: no first-line indent
        after = table._tbl.getnext()
        if after is not None and after.tag == qn("w:p"):
            p_pr = after.get_or_add_pPr()
            ind = p_pr.find(qn("w:ind"))
            if ind is None:
                ind = OxmlElement("w:ind")
                p_pr.append(ind)
            ind.set(qn("w:firstLine"), "0")

    # page numbers, centred at the foot of every page
    for section in doc.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        run = OxmlElement("w:r")
        text = OxmlElement("w:t")
        text.text = "1"
        run.append(text)
        field.append(run)
        footer._p.append(field)
    doc.save(path)


def build(out: Path = OUT) -> Path:
    import pypandoc
    pandoc = pypandoc.get_pandoc_path()
    work = out.parent
    work.mkdir(parents=True, exist_ok=True)
    doc, meta = flattened()
    (work / "paper_flat.tex").write_text(doc, encoding="utf-8")
    (work / "meta.yaml").write_text(meta, encoding="utf-8")
    reference_docx(pandoc, work / "reference.docx")
    resources = ";".join(str(p) for p in (REPO / "paper" / "figures", WORD))
    subprocess.run([pandoc, str(work / "paper_flat.tex"), "-f", "latex", "-t", "docx", "-o", str(out),
                    "--citeproc", "--bibliography", str(FULL / "references.bib"),
                    "--csl", str(WORD / "apa.csl"), "--number-sections",
                    "--metadata-file", str(work / "meta.yaml"),
                    "--reference-doc", str(work / "reference.docx"),
                    "--resource-path", resources], check=True)
    finish(out)
    return out


if __name__ == "__main__":
    path = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print("written", path)
