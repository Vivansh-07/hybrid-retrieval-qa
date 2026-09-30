# CLAUDE.md — hybrid-retrieval-qa

Context file for AI assistants (and future-me) working on this project. Read this first.
Last updated: 2026-09-25.

---

## 1. What this project is

**Title:** Hybrid Sparse–Dense Retrieval with Cross-Encoder Reranking for Domain-Specific Question Answering
**Owner:** Vivansh Garg — B.Tech CSE (Data Science), 3rd year
**Context:** Deep Learning course assignment. Deliverables are a **6–8 page research paper**, a **10–12 slide presentation**, and a **viva**. Also meant to be a resume/portfolio project.
**Timeline:** Started 2026-09-13 with a 4-week plan → paper/viva due roughly **mid-October 2026**.
**Hardware:** Local GPU on Vivansh's Windows PC (code also runs on CPU, just slower).
**Code location:** `D:\Deep Learning\Claude\hybrid-retrieval-qa` (repo name `hybrid-retrieval-qa`).

**Research question:** Which retrieval strategy helps which kind of question, and what does each stage cost?

**Core contribution:** a controlled four-way ablation over one fixed corpus, chunking and index —
`bm25` → `dense` → `fusion` (RRF) → `fusion+rerank` (cross-encoder) — so retrieval strategy is the
only variable. Plus a manual failure analysis and separate scoring of retrieval vs. answer quality.

**Domain / dataset:** Scientific papers — **SciFact (BEIR)**: 5,183 abstracts, 300 test claims,
~1.1 relevant docs per query, document-level relevance judgements.

---

## 2. Pipeline

```
query
  ├── BM25 (rank_bm25, inverted index) ──┐   exact terms, codes, rare entities
  │                                      ├──► Reciprocal Rank Fusion (k = 60)
  └── Dense bi-encoder + FAISS ──────────┘   paraphrase / semantic similarity
                                                   │
                                                   ▼
                                   Cross-encoder rerank (top-50 fused candidates)
                                                   │
                                                   ▼
                              Grounded answer with [n] citations (flan-t5 / roberta-squad2)
```

### Default configuration (`configs/scifact.yaml` — the single source of every hyperparameter)

| Stage | Setting |
|---|---|
| Chunking | `sentence_window`, 3 sentences, stride 2 (1-sentence overlap), min 120 chars; title prepended to every chunk. Alt: `none` (whole abstract) |
| BM25 | k1 = 0.9, b = 0.4 (Anserini/BEIR defaults), lowercase, small built-in stopword list, Porter stemming, top 100 |
| Dense | `sentence-transformers/all-MiniLM-L6-v2`, normalized embeddings, FAISS `IndexFlatIP` (exact), max_seq 256, top 100. Stronger alt: `BAAI/bge-small-en-v1.5` (needs query prefix) |
| Fusion | `rrf`, k = 60, weights [1.0, 1.0], top 100. Alt: `nsf` (min-max normalized score fusion / convex combination) |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2`, candidates = 50, max_len 512, top 10 |
| Answer | `generative` with `google/flan-t5-base`, 5 passages, 128 new tokens. Alt: `extractive` with `deepset/roberta-base-squad2` |
| Eval | k ∈ {1, 3, 5, 10, 20, 100}, primary k = 10, seed = 42 |

---

## 3. Repository layout

```
configs/scifact.yaml        all hyperparameters
src/hybridqa/
  config.py                 YAML → Config dict; dotted lookup (get_path), single-key override()
  data.py                   BEIR loading (HF Hub BeIR/scifact + BeIR/scifact-qrels, or local jsonl),
                            regex sentence splitter, sentence-window chunking, chunk_to_doc()
  sparse.py                 BM25Retriever + tokenize()
  dense.py                  DenseRetriever; embeddings cached to data/indexes/<model>.npy
  fusion.py                 reciprocal_rank_fusion(), normalized_score_fusion(), fuse() dispatch
  rerank.py                 CrossEncoderReranker (reranks head, keeps tail so Recall@100 is comparable)
  answer.py                 AnswerGenerator, faithfulness_signals(), citation_accuracy()
  metrics.py                Recall@k, P@k, MRR, nDCG@k from definitions; paired_bootstrap();
                            validate_against_pytrec_eval() (optional)
  pipeline.py               RetrievalPipeline: builds indexes once, serves all 4 VARIANTS, saves results
  cli.py                    python -m hybridqa.cli {ablation, query, answer}
scripts/
  run_ablation.py           4 variants + significance + --sweep-depth + --sweep-fusion-k
  make_plots.py             results → paper/figures/{ablation_ndcg, recall_curve, latency_accuracy}.png
  failure_analysis.py       picks top-N hard queries → results/failure_analysis.csv for manual labelling
tests/
  test_metrics.py           hand-worked metric checks
  test_fusion.py            RRF formula / scale invariance / weights, chunking, id mapping
paper/
  main.tex                  two-column LaTeX skeleton with TODOs where numbers go
  references.bib            15 entries (BM25, RRF, DPR, BEIR, SciFact, SBERT, monoBERT, monoT5,
                            RAG, FiD, Contriever, SPLADE, ALCE, nDCG, ColBERT)
  Literature_Survey.docx    finished literature survey (see §6)
  README.md                 build instructions + order to write sections
docs/
  WORKFLOW.md               4-week day-by-day plan
  VIVA_PREP.md              expected viva questions in 3 tiers + code questions, with answers
  EXPERIMENT_LOG.md         one entry per run (template inside) — currently no entries filled
  GITHUB_SETUP.md           how to push to GitHub (PAT, not password)
  GIT_FOR_BEGINNERS.md      git from zero: setup, daily routine, common errors
init_and_push.bat / .sh     one-shot git init + first commit + push (usage: init_and_push.bat <github-username>)
data/  results/             gitignored/regenerable (only .gitkeep committed)
```

---

## 4. Commands

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"

python -m pytest tests/ -q                                        # unit tests

python -m hybridqa.cli ablation --variants bm25                   # one variant
python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100   # full run
python scripts/make_plots.py                                      # figures
python scripts/failure_analysis.py --top-n 40                     # annotation worksheet
python -m hybridqa.cli answer --limit 50                          # cited answers → results/answers.json
python -m hybridqa.cli query "Does vitamin D reduce respiratory infections?" --answer

cd paper && latexmk -pdf main.tex                                 # build paper (or Overleaf)
```

### Outputs written to `results/`
`metrics.json`, `ablation.csv`, `significance.json` (paired bootstrap on nDCG@10, fusion+rerank vs each
baseline), `sweep_rerank_depth.json`, `sweep_rrf_k.json`, `runs/<variant>.json` (per-query),
`failure_analysis.csv`, `answers.json`.

---

## 5. Design decisions (defend these in the viva)

- **Rank fusion, not score fusion.** BM25 scores are unbounded and corpus-dependent; cosine is bounded.
  RRF (1/(k+rank)) is scale-invariant and needs no training data. `nsf` is implemented as the baseline
  to show the difference; Bruch et al. (2023) find convex combination can beat RRF *if* labelled data
  exists to tune weights — we are zero-shot, so RRF is the default.
- **Cross-encoder only on a shallow pool.** One forward pass per query–passage pair; `rerank.candidates`
  is the accuracy/latency dial the paper sweeps (10/25/50/100).
- **Chunks are scored, documents are judged.** Chunk rankings collapse to parent doc ids (deduplicated)
  before evaluation, so overlapping chunks cannot inflate metrics.
- **Metrics implemented from definitions**, unit-tested by hand, optionally cross-checked with pytrec_eval.
- **nDCG@10 is the headline metric** — graded and rank-sensitive, which is what reranking changes.
- **Every headline claim gets a p-value** (paired bootstrap, 10k samples, seed 42). With 300 queries,
  +0.01 nDCG is often noise.
- **Reranker keeps the tail** below the rerank depth in original order, so Recall@100 stays comparable.
- **Expected finding:** BM25 may beat general-purpose MiniLM dense retrieval on SciFact. That is a
  *finding* consistent with BEIR, not a bug. If a science-tuned encoder is tried, report both.

### Evaluation plan
- Retrieval: Recall@k, Precision@k, MRR, nDCG@k + latency (ms/query), index build time, embedding size.
- Answers (50 queries, hand-scored 0–2, rubric written first): correctness, completeness, faithfulness,
  citation accuracy. Automatic proxies (`has_citation`, `lexical_overlap`, `abstained`, citation
  precision/recall vs qrels) are reported but are not a substitute.
- Query-subset analysis: split queries by computable property (contains numeral/rare term vs not) and
  compare BM25 vs dense on each — the "which retriever helps which question" result.
- Failure analysis: 40 hardest queries hand-labelled into six fixed categories —
  `terminology_mismatch`, `ambiguous_query`, `missing_evidence`, `conflicting_evidence`,
  `granularity`, `reranker_error`.

---

## 6. Work completed so far

| Date | Done |
|---|---|
| 2026-09-13 | Project scoped: domain = scientific papers, dataset = SciFact (BEIR), local GPU |
| 2026-09-13 | Full codebase scaffolded (all modules, scripts, tests, config) in `D:\Deep Learning\Claude\hybrid-retrieval-qa` |
| 2026-09-13 | Paper skeleton (`main.tex`) + `references.bib`; docs: WORKFLOW, VIVA_PREP, EXPERIMENT_LOG, GITHUB_SETUP |
| 2026-09-13 | 4-week workflow plan written (also saved in the claude.ai Project as `claude/workflow-4-week-plan.md`) |
| 2026-09-13 | **Literature survey** finished — `paper/Literature_Survey.docx` (also in the Project as `claude/literature-survey.md`). 15 refs, 9 sections, comparative table, research gap |
| 2026-09-13 | `GIT_FOR_BEGINNERS.md` + `init_and_push.bat/.sh` added; Vivansh prefers to create the GitHub repo and push himself (no tokens shared with Claude) |

### Literature survey — key points
Five threads: sparse (BM25, SPLADE), dense (SBERT, DPR, Contriever, ColBERT), domain shift (BEIR, SciFact),
fusion (RRF — Cormack 2009; fusion-function analysis — Bruch 2023), reranking (monoBERT, monoT5),
plus RAG/FiD and attribution (ALCE). **Three research gaps** this project addresses:
1. Sparse vs dense is reported as an aggregate — little query-level analysis of *where* they disagree.
2. Fusion and reranking are rarely isolated — whole pipelines are compared, contributions not separable.
3. Cost is reported inconsistently — accuracy at one rerank depth, no cost curve.

### Current status (as of 2026-09-25, end of day)
- **Git/GitHub:** repo initialised; private repo https://github.com/Vivansh-07/hybrid-retrieval-qa
  (`gh` CLI is logged in as Vivansh-07). Make public after grading.
- **Environment:** project venv at `.venv\` on D: (torch 2.6.0+cu126, CUDA works on the RTX 3050
  4 GB). `pip install -e .` done (`pyproject.toml` added). See `docs/RUN_AND_PRESENT.md`.
- **Pipeline verified end to end** on 2026-09-25 (tests, all 4 variants, sweeps, plots, failure CSV,
  answers, Gradio demo), then all test outputs were deleted so Vivansh generates every reported
  number with their own run. `results/` is empty again; `data/hf_cache/` keeps the downloads.
- **Fixes made 2026-09-25:** reranker output was truncated to `rerank.top_k`=10 before evaluation
  (now keeps the full list); `answer.py` no longer uses transformers pipelines (removed in v5);
  `python -m hybridqa.cli` works via `pyproject.toml`; paper converted to IEEEtran.
- **Open decision:** flan-t5-base mostly outputs only a citation (`[3].`) instead of an answer.
  Options (keep + report / extractive mode / flan-t5-large) are in `docs/RUN_AND_PRESENT.md` §5.
- `EXPERIMENT_LOG.md` still has no entries; paper `TODO`s left: abstract numbers, institution.
- Day-by-day plan to 15 Oct (one real commit per day) is in `docs/RUN_AND_PRESENT.md` §4.
- **2026-09-28:** 26–27 Sep tasks were not done (no runs yet); plan revised. Added `--set`/`--out`
  overrides, per-config embedding cache, `dense_from_cache` build stat, `tests/test_config.py`
  (25 tests pass). Still to write: a query-subset analysis script (needed by 6 Oct).
- **2026-09-28 (evening):** Vivansh ran BM25 + the full ablation; results committed (a866a59).
  Headline: fusion significantly beats BM25 and dense; dense vs BM25 and rerank vs fusion are
  NOT significant (see `docs/EXPERIMENT_LOG.md`). Latency varies up to ~2x between runs.
- **2026-09-29:** `hybridqa/significance.py` tests all 6 variant pairs (also better/worse/same
  counts); `scripts/significance.py` recomputes it from `results/runs/` in seconds. Figures
  redesigned for print (column width, greyscale-safe; depth figure now a 2-panel `figure*`).
  Cold-cache dense build: 129.6 s (`results/index_build_timing/`). Log filled for all runs.
  31 tests pass.
- **2026-09-30:** chunking study done (`results/set_chunking.strategy=none/`): whole abstracts do
  not change BM25, but significantly hurt dense (-0.031, p = 0.009) and fusion+rerank (-0.029,
  p = 0.007); 71.6% of whole abstracts exceed the 256-token encoder limit. `scripts/compare_runs.py`
  compares a `--set` run with the main results. **Caveat for the paper:** with sentence windows
  the top-100 passages cover only ~47-58 documents per query, so the main table's "Recall@100"
  is recall within the top-100 passages; @10 metrics are unaffected. 32 tests pass.
  Next: `--variants fusion --set fusion.method=nsf` (1 Oct).
- **2026-09-30 (later):** fusion-method run done early (`results/set_fusion.method=nsf/`): score
  fusion 0.717 vs RRF 0.710, p = 0.27, NOT significant, so never claim RRF beats score fusion;
  RRF is the default for robustness (no normalisation, k-insensitive). `docs/VIVA_PREP.md`
  now holds the real numbers. Found: fusion is chunk-level, so votes for different chunks of one
  gold paper do not combine (claim 1259). None of the side experiments raised accuracy over
  the default config; the project's claim is the comparison, not a tuned score.
  Next: paper section 3 (Method); optional stronger dense encoder (report both).

---

## 7. Next steps (from WORKFLOW.md)

- **Week 1:** install + verify CUDA → load SciFact (check 5,183 / 300) → run tests → random-ranking
  sanity check → BM25 (expect Recall@10 ≈ 0.6–0.8) → dense → chunking study (`none` vs `sentence_window`)
  → write paper §3 Method.
- **Week 2:** hand-check RRF → fusion vs `nsf` → RRF-k sweep → fusion+rerank → depth sweep → full
  `run_ablation.py` → `make_plots.py` → write §4 Setup and §5 Results.
- **Week 3:** failure CSV + hand annotation (2 h) → query-subset analysis → generate & hand-score 50
  answers → write §6 Failure Analysis, §1 Intro, §2 Related Work, §7 Limitations.
- **Week 4:** abstract/conclusion → fix figures → fresh-clone reproducibility pass → slides → viva prep → buffer.

**Slide plan (12):** Title · Problem + failing example · Why BM25 fails · Why dense fails · Pipeline
diagram · RRF equation · Cross- vs bi-encoder · Ablation table · Accuracy-vs-latency plot · Failure
categories · Limitations · Conclusion.

**If behind, cut in this order** (and say so in Limitations): RRF-k sweep → answer annotation (to 25,
not 0) → extractive mode → query-subset analysis.
**Never cut:** the four-row ablation, the failure analysis, the significance test.

---

## 8. Working rules

- **One config change per run.** Changing two things at once teaches nothing about either.
- **Log every run** in `docs/EXPERIMENT_LOG.md` immediately, including failures.
- **Commit every day the code is touched** — `git log` is evidence of own work.
- **Never regenerate a number already in the paper** without re-checking the paper. Leave `evaluation.seed` at 42.
- **Use `--set key=value` for one-change runs** (e.g. `--set chunking.strategy=none`); they write to
  `results/set_<key>=<value>/` and never overwrite the main `results/`. The embedding cache in
  `data/indexes/` is keyed by dataset + chunking + encoder settings, so it no longer needs deleting.
- **Do not "fix" BM25 beating dense** by silently swapping models — report both.
- **Cite nothing that can't be explained in one sentence** — the examiner may ask about any reference.
- **Failure annotation is manual** — do not automate the category labelling.
- Figures must be readable at print size and greyscale-safe.

## 9. Notes for Claude

- Vivansh is a 3rd-year student learning this material for a viva: explain *why*, not just *what*,
  and keep code readable over clever.
- Don't ask for GitHub tokens. The `gh` CLI is already authenticated; commit/push only when asked.
- **Never install or cache anything on C: — the C: drive is nearly full.** Packages go in `.venv\`
  (D:), downloads in `data\hf_cache\` (set in `src/hybridqa/__init__.py`), pip cache in
  `data\pip-cache\` (`.venv\pip.ini`). For pip installs also set `TMP`/`TEMP` to a folder on D:.
  Never `pip install --user` or install into the global Python.
- Vivansh runs the code and presents the output personally; GitHub is only a reference. Prefer
  giving tested commands to run over running experiments for them.
- Windows paths and `cmd`/`.bat` workflows are the default environment.
- Real numbers come only from `results/` — never invent or estimate metrics for the paper or README.
