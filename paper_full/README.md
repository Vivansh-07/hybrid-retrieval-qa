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
   `\newcommand{\onlychapter}{01_introduction}`, then **Recompile**: only that chapter and the
   references it cites are built. References to other chapters show as **??**; that is expected.
   GitHub builds the whole paper on every pull request (above), and checks that this one-chapter
   mode still compiles.
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

So a rewrite in your own words that changes a number, drops a citation's entry or breaks one of
the course's rules fails on the pull request, before it reaches the PDF.

## Before submitting

Every chapter is marked *Drafted with AI assistance*, and the Acknowledgment says so. Rewrite each
chapter in your own words, delete its `% Drafted with AI assistance` comment line, and keep the
Acknowledgment accurate about what was AI-assisted.
