# Paper

`main.tex` is the full paper in **Springer LNCS** format (`llncs` class, `splncs04`
references). Both ship with TeX Live and Overleaf, so no extra files are needed. GitHub
also builds the PDF on every pull request that changes `paper/` (see
`docs/RUN_AND_PRESENT.md`, Step 9).

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Windows with MiKTeX, `latexmk -pdf main.tex` does the same in one command.
Overleaf works too — upload `main.tex`, `references.bib` and `figures/`.

## Order to write it in

1. **Method** — you can write this the moment the code runs; it describes what you built.
2. **Experimental Setup** — exact model names, hardware, metric definitions.
3. **Results** — paste from `results/ablation.csv`, then write the prose around the table.
4. **Failure Analysis** — after you annotate `results/failure_analysis.csv` by hand.
5. **Introduction** — much easier once you know what the results actually say.
6. **Related Work** — read the abstracts of everything in `references.bib`; cite nothing you
   have not at least skimmed, because a viva examiner may ask about any of them.
7. **Abstract and Conclusion** — last, always.

## Length

The course allows Springer (12 pages) or IEEE (strictly 6). The paper uses Springer:
the body ends on page 12 and the references follow on pages 13–14 (8 Oct build). If your
teacher counts references inside the 12 pages, about two more pages have to go; the
header of `main.tex` says what to cut first.
