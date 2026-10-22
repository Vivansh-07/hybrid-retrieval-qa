# The full paper (about 40 pages, APA 7)

The course's general research paper: one improvement to one part of the problem (evidence
retrieval for scientific claims), with a complete methodology, every algorithm as pseudocode,
every equation with its symbols and working, and 40 to 50 references in APA 7. The 12-page
Springer version is in [`../paper/`](../paper/).

| File | What it holds |
|---|---|
| `main.tex` | Preamble (A4, 12 pt, 1.5 spacing, biblatex-apa with biber), title, chapter list |
| `sections/00_abstract.tex` … `08_limitations_conclusion.tex` | One file per chapter |
| `references.bib` | 50 references; authors, venues, pages and DOIs from Crossref or arXiv, the rest hand-checked |
| figures | shared with the short paper, in `../paper/figures/` |

## Getting the PDF

GitHub builds it on every pull request that touches `paper_full/`: the run's **paper_full**
artifact holds `main.pdf` (Actions → paper_full → the run → Artifacts), or in a terminal
`gh run download <run id> -n paper_full`.

## Editing in Overleaf

1. Make the upload from the repository folder (Command Prompt, Windows' built-in `tar`):
   ```
   tar -a -c -f paper_full_overleaf.zip -C paper_full main.tex references.bib sections -C ../paper figures
   ```
   The zip holds the paper, its chapters, the bibliography and a `figures/` folder; the
   paper finds the figures there as well as in `../paper/figures/`.
2. In Overleaf: **New Project → Upload Project** → the zip. Overleaf runs biber for the APA
   references by itself.
3. **Compile one chapter at a time.** Overleaf's free plan stops a compile after 10 seconds, and
   the whole paper takes longer (it times out). In `main.tex`, put the chapter you are working on
   between the braces of `\newcommand{\onlychapter}{}`, for example
   `\newcommand{\onlychapter}{01_introduction}` (line 46), then **Recompile**: only that chapter
   is built, as a quick preview without biblatex and biber (about 3 s from scratch, 1 s per
   recompile, measured on GitHub). In the preview, citations show their keys, e.g.
   `(robertson2009bm25)`, and references to other chapters show as **??**; both are expected.
   GitHub builds the whole paper with its APA references on every pull request (above), and
   checks that the preview still compiles.
4. To bring a finished chapter back: **Menu → Download → Source**, copy the changed chapter files
   over `paper_full/sections/`, and open a pull request. If you copy `main.tex` too, set
   `\onlychapter` back to empty first: `test_paper_full` fails otherwise. The zip is git-ignored.

## What the tests check (`tests/test_paper_full.py`, `tests/test_numbers_full.py`)

- every chapter file exists and is included in order, `\onlychapter` is empty (the whole paper
  is built), every `\ref` has a `\label`, every citation is in `references.bib`, every figure
  exists, and the braces balance;
- every worked example in the Methodology computes to the number it shows
  (`tests/test_worked_examples.py`);
- every headline number (SciFact and NFCorpus) is the one in `results/`, as the text rounds it;
- the abstract has fewer than 200 words, and the bibliography holds 40 to 50 entries.

So a rewrite in your own words that loses a number everywhere it appears, drops a citation's entry
or breaks one of the course's rules fails on the pull request, before it reaches the PDF. A number
is looked for in the whole paper, not chapter by chapter: a mistyped number in one chapter still
passes when the right one appears in another, so check the numbers of a rewritten chapter by eye.

## Before submitting

Every chapter is marked *Drafted with AI assistance*, and the Acknowledgment says so. Rewrite each
chapter in your own words, delete its `% Drafted with AI assistance` comment line, and keep the
Acknowledgment accurate about what was AI-assisted. For the Introduction,
[`briefs/01_introduction.md`](briefs/01_introduction.md) lists what the rewrite has to keep: the
labels and RQ numbers other chapters rely on, the citations, and every number with its source.

`python scripts/rewrite_progress.py` lists every chapter in the suggested order with its marker,
its last commit and its **AI wording**: the share of its five-word phrases that also occur in the
AI draft (commit `16ed095`, the last one before the first rewrite). For a long chapter,
`python scripts/rewrite_progress.py 03_methodology` (or `3`, or `methodology`) shows the same per
section and subsection, and names the first one still at 50% or more, so you can rewrite it and
open a pull request a section at a time. 100% is the draft unchanged, and a light edit stays
high. A low share shows that the draft's wording is gone; it does not by itself prove that the
words are your own, so a chapter counts as rewritten only when you delete its marker. In a
shallow clone the column shows `-` until `git fetch --unshallow` brings in the draft.

`python scripts/overlap_check.py` is a plagiarism pre-check. It lists every run of 8 or more words
that a chapter shares with the short paper, the literature survey or the problem statement, and,
with `--abstracts` (needs internet), with the abstracts of the references. Long shared runs are
what a checker such as Turnitin flags; the survey and the short paper are sources too if they were
submitted to one. A flagged run may be a cited quotation or a name: read each one.
