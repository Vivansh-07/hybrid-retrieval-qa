"""Builds docs/Problem_Statement.pdf, the project explained in plain language.

    python scripts/make_problem_statement.py
"""
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUT = Path(__file__).resolve().parents[1] / "docs" / "Problem_Statement.pdf"

# Plain characters instead of HTML entities; all four exist in the built-in PDF fonts.
_CHARS = {"&ldquo;": "“", "&rdquo;": "”", "&ndash;": "–", "&middot;": "·"}


def _plain(text: str) -> str:
    for entity, char in _CHARS.items():
        text = text.replace(entity, char)
    return text


_Paragraph = Paragraph


def Paragraph(text, style):  # noqa: N802 - same call signature as reportlab's Paragraph
    return _Paragraph(_plain(text), style)

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
ACCENT = colors.HexColor("#2563eb")
SOFT = colors.HexColor("#eef2ff")
LINE = colors.HexColor("#cbd2d9")

body = ParagraphStyle("body", fontName="Helvetica", fontSize=10.2, leading=14, textColor=INK, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=9.5, leading=13, spaceAfter=0)
cell = ParagraphStyle("cell", parent=body, fontSize=9.5, leading=12.5, spaceAfter=0)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=4)
title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=24, textColor=INK,
                       alignment=TA_CENTER, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED,
                          alignment=TA_CENTER, spaceAfter=2)
callout = ParagraphStyle("callout", parent=body, fontSize=11.5, leading=16, spaceAfter=0)


def box(paragraph, bg=SOFT, border=ACCENT):
    t = Table([[paragraph]], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def bullets(items, numbered=False):
    return ListFlowable(
        [ListItem(Paragraph(text, body), leftIndent=14) for text in items],
        bulletType="1" if numbered else "bullet", start="1" if numbered else None,
        bulletFontName="Helvetica-Bold" if numbered else "Helvetica", bulletFontSize=10,
        bulletColor=ACCENT, leftIndent=14)


def grid(rows, widths, header=True):
    data = [[Paragraph(c, cell_b if (header and i == 0) else cell) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return t


def pipeline_diagram():
    """Claim -> (keyword search, meaning search) -> fusion -> reranker -> cited answer."""
    w, h = 17 * cm, 4.2 * cm
    d = Drawing(w, h)

    def node(x, y, bw, bh, lines, fill=SOFT, stroke=ACCENT):
        d.add(Rect(x, y, bw, bh, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1))
        n = len(lines)
        for i, text in enumerate(lines):
            bold = i == 0
            d.add(String(x + bw / 2, y + bh / 2 + (n - 1) * 5.5 - i * 11 - 3, text, textAnchor="middle",
                         fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=8.5 if bold else 7.5,
                         fillColor=INK if bold else MUTED))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1))
        # small arrow head pointing right
        d.add(Polygon([x2, y2, x2 - 5, y2 + 3, x2 - 5, y2 - 3], fillColor=MUTED, strokeColor=MUTED))

    mid = h / 2
    node(0, mid - 16, 62, 32, ["Claim", "(question)"], fill=colors.white, stroke=INK)
    node(88, mid + 8, 104, 34, ["Keyword search", "BM25"])
    node(88, mid - 42, 104, 34, ["Meaning search", "neural (Transformer)"])
    node(218, mid - 16, 72, 32, ["Fusion", "combine lists"])
    node(316, mid - 16, 72, 32, ["Reranker", "neural, re-orders"])
    node(414, mid - 16, 68, 32, ["Answer", "with citations"], fill=colors.white, stroke=INK)

    arrow(62, mid, 75, mid)
    d.add(Line(75, mid + 25, 75, mid - 25, strokeColor=MUTED, strokeWidth=1))
    arrow(75, mid + 25, 88, mid + 25)
    arrow(75, mid - 25, 88, mid - 25)
    d.add(Line(192, mid + 25, 205, mid + 25, strokeColor=MUTED, strokeWidth=1))
    d.add(Line(192, mid - 25, 205, mid - 25, strokeColor=MUTED, strokeWidth=1))
    d.add(Line(205, mid + 25, 205, mid - 25, strokeColor=MUTED, strokeWidth=1))
    arrow(205, mid, 218, mid)
    arrow(290, mid, 316, mid)
    arrow(388, mid, 414, mid)
    return d


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Problem Statement  |  Improving Evidence Retrieval for Scientific Claim Verification")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


story = [
    Paragraph("Improving Evidence Retrieval for<br/>Scientific Claim Verification", title),
    Paragraph("Problem Statement &middot; Deep Learning Course Project", subtitle),
    Paragraph("Vivansh Garg &middot; B.Tech CSE (Data Science), 3rd year &middot; 1 Oct 2026, updated 13 Oct 2026", subtitle),
    Spacer(1, 10),

    Paragraph("1. The project in one sentence", h1),
    box(Paragraph(
        "Given a <b>scientific claim</b>, find the <b>research papers</b> that support or refute it. This "
        "project takes <b>one step</b> of that task, <b>retrieving the evidence</b>, makes it more accurate "
        "and cheaper at the same time, and <b>proves why</b> the improvement works.", callout)),

    Paragraph("2. Why this is needed", h1),
    bullets([
        "There are millions of research papers. Checking a single claim, such as <i>&ldquo;Does vitamin D "
        "reduce respiratory infections?&rdquo;</i>, by hand means searching and reading many abstracts.",
        "AI chatbots answer fluently, but they can <b>make up facts</b>. The safer approach, called "
        "<b>Retrieval-Augmented Generation (RAG)</b>, first finds real documents and then answers only from "
        "them, citing its sources.",
        "Such a system is only as good as its <b>search step</b>: if the right paper is never found, no "
        "answer model can fix it. So finding the evidence is the core problem.",
        "Tools such as <b>SciSpace</b> and <b>Consensus</b> already answer research questions with citations, "
        "and <b>ResearchRabbit</b> maps related papers. But they are closed products: they do not publish how "
        "they find the evidence, how often they miss it, or what each step costs. All of them depend on the "
        "search step, and that is the step this project improves.",
    ]),

    Paragraph("3. The problem", h1),
    Paragraph("There are two main ways to search a collection of papers, and each fails in a different way:", body),
    grid([
        ["Method", "How it works", "Good at", "Fails when"],
        ["<b>Keyword search</b> (BM25)", "Counts the words a claim and a paper share; rare words count more.",
         "Exact terms: gene and drug names, numbers.",
         "The paper uses different words, e.g. &ldquo;heart attack&rdquo; vs &ldquo;myocardial infarction&rdquo;."],
        ["<b>Meaning search</b> (dense neural retrieval)",
         "A Transformer turns each text into a vector of numbers; texts with similar meaning get nearby vectors.",
         "Paraphrases and synonyms.",
         "Rare technical terms, and text unlike what it was trained on (general web text, not science)."],
    ], [3.2 * cm, 5.1 * cm, 3.7 * cm, 5.0 * cm]),
    Spacer(1, 8),
    Paragraph(
        "The standard fix is to <b>combine both</b> (a &ldquo;hybrid&rdquo;) with equal weight and add a "
        "neural <b>reranker</b> that re-orders the best results. The reranker is by far the slowest stage, "
        "and such systems are usually tested as a whole, so it stays unclear <b>which part really helps "
        "and whether the extra computation is worth it</b>.", body),
    box(Paragraph(
        "<b>Problem statement.</b> Can the evidence-retrieval step for scientific claims be made <b>more "
        "accurate and cheaper at the same time</b>, without a reranker? Can we <b>explain</b>, not only "
        "measure, why the improvement works, and does it hold on a <b>second dataset</b>?", callout)),

    KeepTogether([
    Paragraph("4. Aim and objectives", h1),
    Paragraph("<b>Aim:</b> improve one step, evidence retrieval, of a search-and-answer system for "
              "scientific claims, and show why the improvement works.", body),
    bullets([
        "Build the <b>standard pipeline</b> as the baseline: keyword search, meaning search, fusion of the two, "
        "a neural reranker, and an answer generator that cites its sources.",
        "Run a <b>controlled comparison</b> (an ablation): same data and same index, only the search "
        "strategy changes, so every difference can be traced to one stage.",
        "Propose <b>one improvement</b>: fine-tune the meaning-search model on the dataset's own training "
        "claims, and give keyword search a small weight in the fusion instead of a reranker.",
        "<b>Prove</b> where that weight should lie: below a <b>break-even weight</b> of 61/160 = 0.381, keyword "
        "search can only re-order what meaning search found.",
        "Measure <b>accuracy and cost together</b>, and check every comparison with a <b>statistical "
        "significance test</b>, so luck is not reported as progress.",
        "Repeat the improvement <b>unchanged on a second dataset</b> (NFCorpus), and study the remaining "
        "failures <b>by hand</b>.",
    ], numbered=True),
    ]),

    KeepTogether([
        Paragraph("5. How the system works", h1),
        pipeline_diagram(),
        Spacer(1, 4),
        Paragraph("A claim goes to both searches at the same time. Their two ranked lists are merged with "
                  "<b>Reciprocal Rank Fusion</b> (papers that rank high in either list rise to the top). A "
                  "<b>reranker</b> then reads the claim and each top paper together and re-orders them. "
                  "Finally, an <b>answer generator</b> writes a short answer that cites the papers it used. "
                  "<b>The improvement changes two things in this picture:</b> the meaning-search model is "
                  "fine-tuned, and keyword search gets a small weight in the fusion (0.3 instead of 1), which "
                  "makes the reranker unnecessary.", body),
    ]),

    Paragraph("6. Data", h1),
    Paragraph("<b>SciFact</b>, a standard benchmark (part of BEIR) built for exactly this task:", body),
    bullets([
        "<b>5,183</b> abstracts of scientific papers to search.",
        "<b>300 test claims</b>, each with its correct evidence paper(s) marked by experts (about one per claim). "
        "Example: <i>&ldquo;Ivermectin is used to treat onchocerciasis.&rdquo;</i>",
        "<b>809 separate training claims</b>, used only to fine-tune the neural model, never for testing.",
    ]),
    Paragraph("<b>NFCorpus</b>, a second biomedical benchmark (also part of BEIR), to check that the improvement "
              "is not specific to SciFact: <b>3,633</b> abstracts and <b>323 test queries</b> written for the "
              "general public, with about 38 judged abstracts per query.", body),

    KeepTogether([
    Paragraph("7. How success is measured", h1),
    grid([
        ["Measure", "What it tells us (in plain words)"],
        ["<b>nDCG@10</b> (main measure)", "Is a correct paper in the top 10 results, and how high? 1.0 means it is "
         "always ranked first."],
        ["<b>Recall@10</b>", "For what share of claims a correct paper appears anywhere in the top 10."],
        ["<b>Speed</b>", "Milliseconds per question on a laptop GPU (NVIDIA RTX 3050, 4 GB)."],
        ["<b>Significance test</b>", "A paired bootstrap test: is an improvement real, or could it be luck?"],
    ], [4.3 * cm, 12.7 * cm]),
    ]),

    KeepTogether([
    Paragraph("8. Where the deep learning is", h1),
    bullets([
        "<b>Meaning search:</b> a Transformer bi-encoder (MiniLM, 22.7 million parameters) that turns text "
        "into 384-number vectors.",
        "<b>Reranker:</b> a Transformer cross-encoder that reads the claim and a paper together.",
        "<b>Answer generator:</b> Flan-T5, a sequence-to-sequence Transformer.",
        "<b>Training:</b> fine-tuning the meaning-search model with a contrastive loss on each dataset's own "
        "training claims or queries, so it learns what evidence looks like in that domain.",
    ]),
    ]),

    Paragraph("9. Scope and deliverables", h1),
    Paragraph("<b>Scope:</b> one step, evidence retrieval, on English scientific abstracts (SciFact, NFCorpus); "
              "pretrained models plus models fine-tuned by us; everything runs on a laptop GPU.", body),
    Paragraph("<b>Deliverables:</b> a working system with a live web demo; a full research paper (about 40 pages, "
              "APA 7, 50 references, pseudocode for every algorithm and every equation worked through); a "
              "shorter Springer LNCS version (12 pages); a literature survey; a 12-slide presentation; the code "
              "on GitHub with a log of every experiment.", body),

    KeepTogether([
        Paragraph("10. Results on SciFact (300 test claims)", h1),
        grid([
            ["Search setup", "nDCG@10"],
            ["Keyword search only (BM25)", "0.656"],
            ["Meaning search only (pretrained MiniLM)", "0.679"],
            ["Both combined (fusion)", "0.710"],
            ["Combined + reranker (the original full system)", "0.715"],
            ["Meaning search, fine-tuned by us", "0.742"],
            ["<b>The improvement: fine-tuned meaning search + keyword search at weight 0.3, no reranker</b>",
             "<b>0.759</b>"],
        ], [13 * cm, 4 * cm]),
        Spacer(1, 6),
        bullets([
            "Combining the two searches is <b>significantly better</b> than either one alone.",
            "The reranker <b>almost triples the time per question</b> without a significant accuracy gain.",
            "The improvement <b>beats the original system</b> (0.759 against 0.715, p = 0.003) at about a third "
            "of its time per question (114 against 329 ms).",
            "Of the 40 hardest misses, 42% are claims whose marked evidence does not actually state them.",
            "The answer generator is right more often than a copy-a-sentence baseline, but less faithful to its "
            "sources.",
        ]),
    ]),

    KeepTogether([
        Paragraph("11. Results on NFCorpus (323 test queries)", h1),
        grid([
            ["Search setup", "nDCG@10", "ms per query"],
            ["Combined + reranker (the original full system)", "0.358", "168"],
            ["<b>The improvement</b> (weight 0.2, chosen the same way), no reranker", "0.350", "29"],
        ], [11 * cm, 3 * cm, 3 * cm]),
        Spacer(1, 6),
        Paragraph("Here the improvement is <b>level</b> with the original system (the difference is not significant, "
                  "p = 0.35) at about a sixth of its time. The weighted fusion carries over unchanged, and its "
                  "weight again falls below the break-even point; fine-tuning helps less, because NFCorpus's "
                  "plain-language queries make weaker training data.", body),
    ]),

    KeepTogether([
        Paragraph("12. What is new", h1),
        Paragraph("Compared with SciSpace, ResearchRabbit and Consensus, and with recent studies of hybrid search "
                  "(2022&ndash;2026), this project:", body),
        bullets([
            "improves <b>the one step they all depend on</b>, evidence retrieval, and makes it cheaper: "
            "fine-tuning plus weighted fusion replaces the reranker;",
            "<b>proves</b> what the fusion weight does: below 61/160 = 0.381, keyword search can only re-order "
            "what meaning search found, which explains the best weight on both datasets;",
            "measures every stage <b>openly</b>, with a significance test for each comparison, on two datasets, "
            "on a laptop and without sending data to a cloud service;",
            "explains <b>claim by claim</b> which search wins: it depends on how many words the claim shares with "
            "its evidence;",
            "measures whether a search score can tell the answer generator <b>when not to answer</b>: only at a "
            "high cost.",
        ]),
    ]),

    KeepTogether([
    Paragraph("13. Key terms", h1),
    grid([
        ["Term", "Meaning"],
        ["BM25", "Classic keyword-matching score used by search engines."],
        ["Dense retrieval / embedding", "Search by meaning: texts become vectors, and nearby vectors mean similar meaning."],
        ["Fusion (RRF)", "Merging two ranked lists by rank position, not by raw score."],
        ["Break-even weight", "The keyword-search weight (61/160 = 0.381) below which keyword search can only "
         "re-order what meaning search found, never add to it."],
        ["Reranker (cross-encoder)", "A slower, more careful model that re-scores only the top results."],
        ["RAG", "Retrieval-Augmented Generation: find documents first, then answer from them."],
        ["Fine-tuning", "Further training of a pretrained model on data from our own domain."],
        ["Ablation", "Removing or changing one part at a time to see what each part contributes."],
        ["nDCG@10", "How well the right papers are ranked in the top 10 (1 = perfect)."],
    ], [4.3 * cm, 12.7 * cm]),
    ]),
]



def build(out: Path = OUT) -> None:
    """Write the PDF to `out` (tests build it into a temporary folder and compare)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.9 * cm,
                            title="Problem Statement: Improving Evidence Retrieval for Scientific Claim Verification",
                            author="Vivansh Garg")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
    print("written", OUT)
