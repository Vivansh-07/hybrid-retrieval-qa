# Four-week workflow

Target: a working pipeline, a 6–8 page paper, a slide deck, and enough
understanding to survive a viva. Dataset is SciFact (BEIR), hardware is your local GPU.

The ordering is deliberate. Most student projects fail the viva not because the
code is wrong but because the numbers arrive in week 4 and there is no time left to
understand them. This plan has every headline number in hand by the end of week 2.

---

## Week 1 — Data, baselines, and a metric you trust

**Goal by Sunday: BM25 and dense numbers on the board, and you believe them.**

| Day | Do this | Done when |
|---|---|---|
| 1 | `pip install -r requirements.txt`, verify CUDA (`python -c "import torch; print(torch.cuda.is_available())"`) | prints `True` |
| 1 | Load SciFact, print corpus size, query count, mean gold docs per query | matches 5,183 / 300 |
| 2 | Read `src/hybridqa/metrics.py` line by line. Run `pytest tests/test_metrics.py -v` | you can compute nDCG@5 on paper for a 5-doc list |
| 2 | Sanity check: evaluate a *random* ranking. Note how low it is | you know what "chance" looks like |
| 3 | `python -m hybridqa.cli ablation --variants bm25` | Recall@10 in the 0.6–0.8 band for SciFact |
| 4 | `python -m hybridqa.cli ablation --variants dense` | dense is close to, maybe below, BM25 — this is expected and it is the paper's premise |
| 5 | Chunking study: run both with `chunking.strategy: none` vs `sentence_window` | you know whether chunking helps here and can say why |
| 6–7 | Write the **Method** section (§3) of the paper. Nothing about results yet | 1.5 pages of LaTeX that compile |

**Checkpoint question to yourself:** if BM25 beats your dense model, is that a bug
or a finding? (On SciFact with a general-purpose MiniLM, it is a finding — BEIR
reports exactly this. Do not "fix" it by silently swapping in a model fine-tuned on
scientific text. If you do swap, report both.)

---

## Week 2 — Fusion, reranking, and the ablation table

**Goal by Sunday: the four-row table is filled in and statistically tested.**

| Day | Do this | Done when |
|---|---|---|
| 8 | Read `fusion.py`. Hand-compute RRF for a 3-document toy case | your arithmetic matches `pytest tests/test_fusion.py` |
| 9 | Run `fusion`. Compare against `normalized_score_fusion` by flipping `fusion.method` to `nsf` | you can say why rank fusion beats score fusion here |
| 9 | `--sweep-fusion-k 10 30 60 100` | you can show the result is not a k artefact |
| 10 | Run `fusion+rerank`. Expect the largest single jump of the project | nDCG@10 up meaningfully over fusion |
| 11 | `--sweep-depth 10 25 50 100` | you have the accuracy-vs-latency curve and know where it saturates |
| 12 | Full run: `python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100` | `results/metrics.json`, `ablation.csv`, `significance.json` all exist |
| 12 | `python scripts/make_plots.py` | three PNGs in `paper/figures/` |
| 13–14 | Write **Experimental Setup** (§4) and the **Results** table + prose (§5) | table has real numbers, no `TODO` left in §4–5 |

**Do not skip the significance test.** With 300 queries, a +0.01 nDCG difference is
often noise. `results/significance.json` gives you a p-value; if fusion's gain over
BM25 is not significant, say so — that is a more interesting paper than pretending.

---

## Week 3 — The parts that make it a research paper

**Goal by Sunday: failure analysis done, answer quality measured, full draft exists.**

| Day | Do this | Done when |
|---|---|---|
| 15 | `python scripts/failure_analysis.py --top-n 40` | CSV of 40 hard queries |
| 15–16 | **Annotate all 40 by hand** into the six fixed categories. Budget 2 hours. Do not automate this | `category` column is complete |
| 17 | Split queries by a computable property (contains a numeral/rare term vs. not) and report BM25 vs dense on each subset | you have the "which retriever helps which question" result — this is the paper's real contribution |
| 18 | `python -m hybridqa.cli answer --limit 50` | `results/answers.json` |
| 18–19 | Score 50 answers by hand on correctness / completeness / faithfulness / citation accuracy. Use a 0–2 scale, write the rubric down first | a small table you can put in §5.2 |
| 20 | Write **Failure Analysis** (§6) with 2–3 verbatim examples | 3/4 page |
| 21 | Write **Introduction** (§1), **Related Work** (§2), **Limitations** (§7) | complete draft, no TODOs |

**On Related Work:** read at least the abstract of every paper in `references.bib`.
An examiner who sees `\cite{khattab2020colbert}` may ask what late interaction is.
Cite nothing you cannot describe in one sentence.

---

## Week 4 — Polish, slides, viva

| Day | Do this |
|---|---|
| 22 | Write the **Abstract** and **Conclusion**. Reread the whole draft aloud — you will hear the bad sentences |
| 23 | Fix figures: readable at print size, greyscale-safe, axis labels with units |
| 24 | Reproducibility pass: `git clone` your own repo into a fresh folder, `pip install -r requirements.txt`, run the ablation end to end. Fix whatever breaks |
| 25 | Build the slide deck: 10–12 slides (see split below) |
| 26 | Read `VIVA_PREP.md`. Answer every question out loud, without notes. Mark the ones you fumble |
| 27 | Re-read the code you wrote three weeks ago. You *will* be asked "what does this line do" |
| 28 | Buffer. Something always breaks |

**Slide split (10–12 slides):**
1. Title 2. The problem, with one concrete failing example 3. Why BM25 alone fails
4. Why dense alone fails 5. Pipeline diagram 6. RRF in one equation
7. Cross-encoder vs bi-encoder, one diagram 8. Ablation table 9. Accuracy-vs-latency plot
10. Failure categories 11. Limitations 12. Conclusion

Slide 5 is the one you will spend the most time on and the one they will look at longest.

---

## Standing rules

- **Commit every day you touch the code.** `git log` is evidence of your own work, and
  a supervisor who suspects otherwise will look at it.
- **Log every run in `EXPERIMENT_LOG.md`** the moment it finishes, including the failed
  ones. "We tried X, it did not help, here is the number" is a paper section.
- **Never regenerate a number you have already put in the paper** without re-checking the
  paper. Set `evaluation.seed` and leave it alone.
- **One config change per run.** If you change the encoder and the chunk size together
  you have learned nothing about either.

## If you fall behind

Cut in this order, and say what you cut in Limitations:
1. The RRF-k sweep (keep k=60, cite Cormack et al.)
2. The answer-quality annotation (drop to 25 answers, not zero)
3. The extractive answering mode (generative only)
4. The query-subset analysis

Never cut: the four-row ablation, the failure analysis, or the significance test.
Those three are what separate a project report from a paper.
