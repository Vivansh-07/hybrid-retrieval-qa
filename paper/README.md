# Paper

`main.tex` is a skeleton, not a draft: every section has a comment describing what
belongs in it, and `TODO` marks every place a number from `results/` has to be pasted.

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

Aim for 6–8 pages in this two-column format. If a section will not fill its space
honestly, the fix is more analysis, not more adjectives.
