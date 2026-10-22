# Rewrite brief: Introduction (`sections/01_introduction.tex`)

What the rewritten Introduction has to keep so the paper and its tests still work (issue #40).
Notes, not text for the paper: write from these in your own words, not from the draft.
Line numbers (l.) are the draft's, at commit d488d70; `02:196` means `sections/02_*.tex` line 196.

## 1. How to work

1. Overleaf, one chapter: `main.tex` line 46 -> `\newcommand{\onlychapter}{01_introduction}`,
   Recompile (upload steps: [`../README.md`](../README.md)).
2. Preview has no biber: citations show their keys (`scispace`, `(lewis2020rag)`), `\ref` to other
   chapters shows `??`; both expected. It defines only one-argument `\parencite`, `\textcite`,
   `\cite` (`main.tex` 57-59): no `[p.~3]` options, no `\citeauthor`.
3. Done: delete l.4 (`% Drafted with AI assistance (Claude). ...`); Download -> Source; copy the
   file over `paper_full/sections/`; if you copy `main.tex` too, `\onlychapter` back to `{}`. One PR.
   Check that the Acknowledgment (`main.tex` 104-106) still says truthfully what AI did.
4. Locally: `python -m pytest tests/test_paper_full.py tests/test_numbers_full.py -q` (29 passed today).
5. `python scripts/rewrite_progress.py 01_introduction`: per section, how much of the AI draft's
   wording is still in your text.

## 2. Unit by unit

Header l.7-8: `\section{Introduction}`, `\label{sec:introduction}`.

**1.1 Background and motivation** (l.10-28, `sec:motivation`)
- assistants: question in -> cited answer out; all RAG (retrieve, then LM writes + cites)
- retrieval miss -> nothing to cite, whatever the LM -> retrieval = ceiling on answers
- retrieval = also the cost (reranker reads every candidate with the query)
- cites: `scispace`, `consensus`, `asai2026openscholar`, `skarlinski2024paperqa`, `lewis2020rag`, `nogueira2019passage`

**1.2 The part of the problem this paper improves** (l.30-48, `sec:problem`)
- scope = one step: claim -> rank abstracts, evidence (supports/refutes) on top
- why this step x3: all assistants need it; measurable (SciFact marks evidence); cost sits here
- baseline: BM25 + bi-encoder -> RRF -> cross-encoder; the change: fine-tune encoder, down-weight
  BM25 (weight tuned on dev claims), drop reranker -> proof in `sec:method`
- example claim (aspirin/PGE2) = SciFact test claim, also 03:73, 06:72, 06:100; optional
- cites: `wadden2020scifact`, `chen2022hybrid`, `bruch2024fusion`

**1.3 Research questions** (l.50-63, `sec:rq`): numbers and topics fixed (Section 3); the wording is
yours. Topics: RQ1 improvement vs standard pipeline, quality and cost / RQ2 the fusion weight, derived
or searched / RQ3 retriever per claim type, source of the fusion gain / RQ4 second dataset /
RQ5 remaining failures, consequence for answers

**1.4 Contributions** (l.65-97, `sec:novelty`): five, contribution n answers RQn.
- C1 the retriever: size, training claims, weight, nDCG@10 vs baseline + p, Recall@10, ms; cheaper = no reranker
- C2 `Proposition~1`: break-even weight 61/160; below it BM25 re-orders dense candidates, adds none -> rule, not search
- C3 word-overlap split: dense ahead on paraphrased (+ p), BM25 holds on high overlap, fusion never below the better
  (pretrained encoder at equal weights only: Section 5)
- C4 NFCorpus: weight again below break-even, level with reranked pipeline (p), 1/6 of the time, fine-tuning helps less; `sec:res-nfcorpus`
- C5 limits: 42% of the 40 hardest not retrieval errors; generative reader: more correct, unsupported in 1/3; reranker score no cheap abstain signal

**1.5 How this work differs** (l.99-133, `sec:tools`, `tab:tools`): heading wording free.
- ResearchRabbit: discovery, no answers; Consensus, SciSpace: cited answers, closed (method, miss
  rate, cost unpublished); OpenScholar, PaperQA2: open, whole-system evaluation only
- this work: their shared retrieval step, per stage, significance-tested; offline (4 GB laptop GPU)
- cites: `researchrabbit`, `consensus`, `scispace`, `asai2026openscholar`, `skarlinski2024paperqa`

**1.6 Organisation of the paper** (l.135-144, no label): one item per chapter 2-8.
Refs: `sec:literature`, `sec:method`, `sec:setup`, `sec:results`, `sec:failures`, `sec:discussion`, `sec:limitations`.

## 3. What other chapters depend on

**Labels**: keep all seven. `sec:introduction` (l.8), `sec:motivation` (11), `sec:problem` (31),
`sec:rq` (51): referenced nowhere. `sec:novelty` (66): from 02:196. `sec:tools` (100): from 02:134.
`tab:tools` (120): from l.113 only. Dropping a referenced one fails `test_every_ref_has_a_label`.

**`\ref` targets used here** (keep each on its chapter): `sec:method` 03:10 (l.47, 139),
`sec:res-nfcorpus` 05:156 (l.90), `sec:literature` 02:7, `sec:setup` 04:7, `sec:results` 05:9,
`sec:failures` 06:9, `sec:discussion` 07:6, `sec:limitations` 08:6.

**RQ1-RQ5** are typed text, not `\ref`: LaTeX will not catch a renumbering.
02:204 RQ1, 209 RQ2, 212 RQ3, 215 RQ5 (gap list); 04:23 RQ4; 05:13-15 (overview), headings
53 RQ1, 104 RQ2, 155 RQ4, 225 RQ3, plus 100 and 217; 06:13 RQ5; 07:11, 20, 28, 34, 44 (paragraph
headings RQ1-RQ5, each a short restatement: your RQs must still match them).

**Proposition 1** is a `\paragraph` at 03:369, not a numbered environment, so `Proposition~1` is
typed text (l.77, 140 and in 02, 03:15, 05, 06, 07, 08). Keep the name and the 1.

**Contributions vs gaps.** 02:194-196: "four gaps ... each corresponds to a contribution"
(`sec:novelty`). Gap 1 reranking cost (RQ1) -> C1; gap 2 weight unexplained (RQ2) -> C2;
gap 3 which retriever, which claim (RQ3) -> C3; gap 4 closed tools, end-to-end only (RQ5) -> C5.
C4 (RQ4) answers no gap. Keep five, numbered as the RQs; merging or reordering means editing
02:194-215 in the same PR.

**Table `tab:tools`** (l.116-133): keep `\label{tab:tools}`, the six rows and the yes/no/partial
values (facts); the caption (l.118-119) is draft text, so write it yourself. "as of October 2026"
(l.114) = `urldate = {2026-10-13}` of the three tools, `references.bib` 450-469: change both or neither.

## 4. Citations

| Key | Draft | In the PDF | Also in |
|---|---|---|---|
| `scispace`, `consensus` | `\textcite`, l.14, 126-127 | SciSpace (n.d.), Consensus (n.d.) | 02 |
| `researchrabbit` | `\textcite`, l.103, 125 | ResearchRabbit (n.d.) | 02 |
| `asai2026openscholar` | `\parencite`, l.16, 128 | (Asai et al., 2026) | 02 |
| `skarlinski2024paperqa` | `\parencite`, l.16, 129 | (Skarlinski et al., 2024) | 02 |
| `lewis2020rag` | `\parencite`, l.19 | (Lewis et al., 2020) | 02 |
| `nogueira2019passage` | `\parencite`, l.28 | (Nogueira & Cho, 2019) | 02, 03 |
| `wadden2020scifact` | `\parencite`, l.38 | (Wadden et al., 2020) | 02, 04 |
| `chen2022hybrid`, `bruch2024fusion` | one `\parencite`, l.44 | both in one parenthesis | chen 02; bruch 02, 03 |

- Tools have no date, hence n.d. Name one with `\textcite{scispace}`, not `SciSpace \parencite{scispace}`
  ("SciSpace (SciSpace, n.d.)"; commit 6061910). The bare name after that is fine (l.104).
- All ten are cited in 02 too: dropping one here keeps its reference.
- New key: add it to `references.bib`. The bib has 50 entries, all cited; the test allows 40-50
  (`tests/test_numbers_full.py` 90-92), so a new key means removing an entry, and none is cited
  only in the Introduction: that also means dropping its citations in another chapter.

## 5. Numbers

`tests/test_numbers_full.py` joins all chapters (l.24) and checks each value appears **somewhere
in the paper** (l.61, 72, 78). Every number below is also in another chapter, so a typo in the
Introduction passes the tests: check this table by eye.
`FT/` = `results/dense_minilm_finetuned__fusion_weighted/`, `NF/` = `results/nfcorpus/`,
`NFT/` = `results/nfcorpus_finetuned__fusion_weighted/`.

| Intro (l.) | Source | Raw | Test | Also in |
|---|---|---|---|---|
| nDCG@10 0.759 (73) | `FT/ablation.csv` fusion | 0.7587 | yes | 00 05 07 08 |
| vs 0.715 (74) | `results/ablation.csv` fusion+rerank | 0.7154 | yes | 00 05 07 08 |
| p = 0.003 (75) | `RESULTS.md` step 11, `docs/EXPERIMENT_LOG.md` 504 | 0.0027 | no | 00 03 05 07 08 |
| Recall@10 0.905 vs 0.833 (75) | `FT/ablation.csv`; `results/ablation.csv` | 0.905, 0.8329 | 0.905 only | 05 07 |
| 114 vs 329 ms (76) | same two files, `latency_ms_per_query` | 113.67, 328.98 | no | 05 07 |
| 689 training claims (73) | `results/finetune_all-MiniLM-L6-v2-scifact/history.json` (809 - 120 dev, 03:265) | 689 | no | 02 03 04 05 06 07 |
| 300 test claims (74) | `configs/scifact.yaml` 5; `num_queries` | 300 | no | 00 03 04 05 08 |
| 22 million parameters (72) | no results/ file; 22.7M in 03:210, 04:82 | 22.7M | no | 02 08 ("22-million"); 03 04 say 22.7: pick one |
| weight 0.3 (73) | `results/fusion_weights_all-MiniLM-L6-v2-scifact.json` `chosen` | 0.3 | checks "w = 0.3" (not in the Intro) | 03 05 07 |
| 61/160, 0.38 (78) | (k+1)/(k+K): k = 60, K = 100, `configs/scifact.yaml` 38, 20, 32 | 0.38125 | no | 00 03 05 07 08 |
| +0.073, p = 0.022 (84) | `results/query_subsets.json` overlap, low, "dense vs bm25" | 0.0733, 0.0215 | no | 05 07 |
| NFCorpus 0.350 vs 0.358, p = 0.35 (88) | `NFT/ablation.csv` fusion; `NF/ablation.csv` fusion+rerank; `results/nfcorpus_comparisons.json` "improved vs standard pipeline" | 0.3502, 0.3584, 0.3494 | yes | 00 05 07 |
| a sixth of its time (89) | same two files, latency | 29.24 vs 167.54 ms | no | 00 05 07 08 |
| NFCorpus weight below break-even (87) | `results/fusion_weights_all-MiniLM-L6-v2-nfcorpus.json` | 0.2 | "w = 0.2" | 05 |
| 40 hardest, 42% (91-92) | `results/failure_summary.json` missing_evidence | 17 of 40 | no | 00 06 07 |
| right more often; a third of its answers (93-95) | `results/answer_quality.json` correctness; `results/answer_scoring.csv` generative with faithfulness 0 (reader: `answer_scoring_key.json`) | 0.84 vs 0.48; 17 of 50 | no | 05 07 |
| reranker confidence (95) | `results/abstention.json` cross_encoder_top | 68/74 misses caught, 109/226 refused | no | 03 05 |
| 4 GB GPU (112) | `docs/EXPERIMENT_LOG.md` "Hardware:" lines (RTX 3050 Laptop) | - | no | 04 |

Rounding: keep what the other chapters show.
- p = 0.022: 0.0215 = 215/10,000 rounds to 0.022 (Python's `f"{0.0215:.3f}"` gives 0.021: float).
- 42%: 17/40 = 42.5%, printed 42% by `scripts/failure_analysis.py` 141 (42/28/20/10 sum to 100).
- C3 is the **pretrained** encoder at equal weights (05:241 says so). With the fine-tuned one,
  fusion is below dense on paraphrased claims (0.563 vs 0.620, `results/dense_minilm_finetuned/query_subsets.json`).

## 6. Claims about other work (citation next to each)

- OpenScholar and PaperQA2 report different results (02:112-117: OpenScholar on citation accuracy,
  PaperQA2 on literature tasks). The draft (l.15-17) credits both with expert-level citations;
  give each system its own result.
- "several times slower" (l.26-28, `nogueira2019passage`): the citation backs the mechanism (02:94-96).
  Own numbers, ms per claim: SciFact `results/ablation.csv` fusion 119.28, fusion+rerank 328.98
  (50 candidates, `configs/scifact.yaml` 46): the rerank step adds 210 = 1.8x the fused retrieval
  (9.6x dense alone, 21.91). NFCorpus `NF/ablation.csv` 31.2 vs 167.54: step 4.4x.
  So "several times" fits NFCorpus, or dense retrieval alone, not SciFact's fused retrieval (1.8x):
  name the comparison.
- Closed tools publish no method, miss rate or cost: their sites as of 2026-10-13 (bib `urldate`; 02:129-133).

## 7. Length

Draft: 1,002 words (`python scripts/rewrite_progress.py`) for "about 4 pages" (l.2). No test
checks it. The Overleaf one-chapter preview gives the Introduction alone: its pages minus the title
page (the **paper_full** run on a PR reports the whole paper's count).
