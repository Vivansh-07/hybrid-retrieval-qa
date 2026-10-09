# Hybrid Sparse–Dense Retrieval for Scientific Question Answering

[![tests](https://github.com/Vivansh-07/hybrid-retrieval-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/Vivansh-07/hybrid-retrieval-qa/actions/workflows/tests.yml)
[![paper](https://github.com/Vivansh-07/hybrid-retrieval-qa/actions/workflows/paper.yml/badge.svg)](https://github.com/Vivansh-07/hybrid-retrieval-qa/actions/workflows/paper.yml)

A multi-stage retrieval pipeline for question answering over a specialised corpus,
evaluated with a controlled ablation on **SciFact** (BEIR).

**[RESULTS.md](RESULTS.md)**: every experiment in order, with its command, its output and the
decision it led to.

```
query
  ├── BM25 over an inverted index ─────────┐
  │     exact terms, codes, rare entities  │
  │                                        ├──► Reciprocal Rank Fusion
  └── Dense bi-encoder + FAISS ────────────┘         (rank-level, k = 60)
        paraphrase, semantic similarity                      │
                                                             ▼
                                              Cross-encoder reranking
                                              (full attention, top-50 only)
                                                             │
                                                             ▼
                                              Grounded answer with citations
```

**The question this project answers:** which retrieval strategy helps which kind of
question, and what does each stage cost? The contribution is the ablation — BM25 alone,
dense alone, fusion, fusion + reranking — over one fixed corpus and index, plus a manual
failure analysis of what still goes wrong.

---

## Quick start

```bash
git clone https://github.com/Vivansh-07/hybrid-retrieval-qa.git
cd hybrid-retrieval-qa
python -m venv .venv
.venv\Scripts\activate                              # Windows (Command Prompt)
# source .venv/bin/activate                          # Linux / macOS
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build
python -m pip install -r requirements.txt
python -m pip install -e .

python -m pytest tests/ -q                           # 73 tests

# full four-way ablation + sweeps (downloads SciFact + models on first run, ~13 min on an RTX 3050)
python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100

# figures for the paper
python scripts/make_plots.py

# one ad-hoc question, with a cited answer
python -m hybridqa.cli query "Does vitamin D supplementation reduce respiratory infections?" --answer

# interactive web demo: four strategies side by side + cited answer (http://127.0.0.1:7860)
python scripts/demo.py
python scripts/demo.py --best                        # the best system (fine-tuned encoder + weighted fusion)
```

Downloads (SciFact, models) are cached inside the project in `data/hf_cache/`, not in
your user profile. **Step-by-step instructions, expected output and a presentation
script:** [docs/RUN_AND_PRESENT.md](docs/RUN_AND_PRESENT.md).

If `torch.cuda.is_available()` is `False`, everything still runs on CPU — expect the
dense indexing step to take a few minutes instead of seconds.

---

## Results

SciFact test set, 300 claims, from `results/ablation.csv` (RTX 3050 Laptop GPU). Latency
is approximate: it varied up to about 2x between identical runs.

| Variant | nDCG@10 | Recall@10 | MRR@10 | ms/query |
|---|---|---|---|---|
| BM25 | 0.656 | 0.784 | 0.623 | 80 |
| Dense (all-MiniLM-L6-v2) | 0.679 | 0.815 | 0.641 | 22 |
| RRF fusion | 0.710 | 0.833 | 0.675 | 119 |
| + cross-encoder rerank | 0.715 | 0.833 | 0.685 | 329 |

Paired bootstrap on nDCG@10 (10,000 resamples, `results/significance.json`):

- Dense vs BM25: not significant (p = 0.25). The two are tied, but succeed on different claims.
- Fusion vs BM25 and vs dense: significant (p < 0.0001 and p = 0.019).
- Reranker vs fusion: not significant (p = 0.65), at about 3x the latency.

With a stronger dense encoder (`BAAI/bge-small-en-v1.5`, `results/dense_bge_small/`):

| Variant | nDCG@10 | vs MiniLM run |
|---|---|---|
| Dense | 0.723 | +0.044, p = 0.002 |
| RRF fusion | 0.721 | +0.012, not significant |
| + rerank (ms-marco-MiniLM) | 0.712 | not significant |
| + rerank (bge-reranker-base) | 0.714 | not significant, about 1.3 s/query |

Here dense clearly beats BM25 and fusion adds nothing over dense alone: the fusion gain
depends on the two retrievers being of similar strength. Neither reranker helps.

With the MiniLM encoder **fine-tuned on SciFact's training claims** (`scripts/finetune_dense.py`:
contrastive loss, 689 training claims, 120 dev claims for early stopping, no test claim seen;
`results/dense_minilm_finetuned/`):

| Variant | nDCG@10 | Recall@10 | vs before fine-tuning |
|---|---|---|---|
| Dense | **0.742** | 0.886 | +0.063, p < 0.0001 |
| RRF fusion | 0.731 | 0.888 | +0.022, p = 0.013 |
| + cross-encoder rerank | 0.725 | 0.848 | not significant |

Fine-tuning is the largest gain of any change tried, at no extra query cost. The 6-layer
fine-tuned model is statistically level with the larger off-the-shelf BGE-small (p = 0.23).

**Best system:** the fine-tuned encoder fused with BM25 at a BM25 weight of 0.3, chosen on
held-out development claims (`scripts/tune_fusion.py`): **nDCG@10 0.759, Recall@10 0.905**,
significantly better than equal-weight fusion (p = 0.0009) and than the original
pipeline (+0.043, p = 0.003). Results in `results/dense_minilm_finetuned__fusion_weighted/`.
Every run is recorded in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

**Failure analysis** of the 40 hardest claims (`results/failure_analysis.csv`): missing
evidence in the gold abstract 42%, terminology mismatch 28%, reranker error 20%, ambiguous
claim 10%. **Answer quality** (100 answers, 0-2 rubric, `results/answer_quality.json`): the
generative reader is more often correct (0.84 vs 0.48) but less faithful (1.16 vs 2.00)
than the extractive one. Both were labelled/scored by an AI assistant following fixed
rubrics (see the paper and `docs/ANSWER_RUBRIC.md`).

**Further checks** (details in the experiment log):
- At a BM25 weight of 0.3, BM25 only re-orders the dense retriever's top 100 and never
  adds a document: a passage found only by BM25 scores at most 0.3/61, below the dense
  retriever's 100th passage (1/160). Only 2 test claims lose their gold abstract this way.
- Could the reader abstain when retrieval failed? The top cross-encoder score predicts
  whether a gold abstract reached the reader with AUC 0.81, but the threshold chosen on
  the dev claims catches 68 of 74 misses only by refusing 109 of 226 answerable claims
  (`scripts/abstention.py`).
- The best system, fine-tuning included, reproduces exactly from a fresh clone.

---

## Repository layout

```
configs/scifact.yaml     every hyperparameter for every stage, in one file
src/hybridqa/
  config.py              YAML loading, dotted-path lookup, single-key override
  data.py                BEIR loading, sentence-window chunking, chunk→doc mapping
  sparse.py              BM25 retriever (rank_bm25) with the project's tokenisation
  dense.py               bi-encoder + FAISS flat inner-product index, cached embeddings
  fusion.py              Reciprocal Rank Fusion, plus a score-fusion baseline
  rerank.py              cross-encoder reranking over the fused candidate pool
  answer.py              extractive / generative answering with citations
  metrics.py             Recall@k, P@k, MRR, nDCG@k, paired bootstrap
  pipeline.py            builds indexes once, serves all four ablation variants
  cli.py                 `python -m hybridqa.cli {ablation,query,answer}`
scripts/
  run_ablation.py        the four variants + rerank-depth and RRF-k sweeps
  make_plots.py          results/*.json → paper/figures/*.png
  failure_analysis.py    the 40 hardest claims as a worksheet for manual labelling; --summarize
  demo.py                Gradio web demo: one question, all four variants side by side; --best
  pick_demo_claims.py    claims worth showing in the demo (where the variants disagree)
  abstention.py          could retrieval confidence tell the reader to abstain? (AUC, threshold)
  finetune_dense.py      contrastive fine-tuning of the dense encoder on SciFact train claims
  significance.py        recompute significance.json from saved runs
  compare_runs.py        paired comparison of a one-change run against the main results
  tune_fusion.py         choose the BM25 fusion weight on held-out dev claims
  query_subsets.py       which retriever helps which kind of claim (overlap, numbers, rare terms)
  answer_sheet.py        blind sheet for hand-scoring the generated answers; --summarize
tests/                   unit tests: metrics, fusion, config, significance, subsets, scoring,
                         abstention; the paper's references and numbers; the review sheet
paper/                   LaTeX source, bibliography, figures (GitHub builds the PDF: Actions → paper)
docs/
  WORKFLOW.md            week-by-week plan through to the viva
  VIVA_PREP.md           expected questions with worked answers
  ANSWER_RUBRIC.md       how the generated answers are scored by hand (fixed in advance)
  RUN_AND_PRESENT.md     tested commands, expected output, presentation script, daily plan
  ALGORITHMS.md          every algorithm: the idea, why it was chosen, settings, code
  LITERATURE_SURVEY.md   the survey (40 references, 2022-2026 update); builds the .docx
  FAILURE_REVIEW.md      the 40 failure labels as a review sheet (generated)
  SLIDES.md              the 12 slides: titles, what to say, where each number comes from
  Problem_Statement.pdf  the project explained in plain language
  EXPERIMENT_LOG.md      append one entry per run, including failed ones
```

---

## Design decisions worth knowing before you present this

- **Fusion happens at rank level, not score level.** BM25 scores are unbounded and
  corpus-dependent; cosine similarities are bounded. No fixed weighting of the raw
  numbers transfers across queries, so RRF combines positions instead. `fusion.py`
  also implements min-max score fusion so the paper can show the difference.
- **The cross-encoder never sees the corpus.** It costs one forward pass per
  query–passage pair, so it runs only over the fused top-`rerank.candidates`. That
  parameter is the accuracy/latency dial the paper measures.
- **Chunks are scored, documents are judged.** BEIR relevance is document-level, so
  chunk rankings are collapsed to parent document ids before evaluation and
  de-duplicated. Overlapping chunks therefore cannot inflate the metrics.
- **Metrics are implemented from their definitions** rather than imported, so every
  number is defensible in a viva. `validate_against_pytrec_eval` cross-checks them
  against the standard toolkit if you install `pytrec_eval`.
- **Every headline claim gets a p-value.** With 300 queries a one-point nDCG difference
  is often noise; `run_ablation.py` writes a paired bootstrap test to
  `results/significance.json`.

---

## Reproducibility

Every number in the paper comes from `results/`, produced by the scripts above, and
every run is recorded in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md). Seeds are
fixed (`evaluation.seed`); on 3 Oct a fresh clone reproduced the best system exactly,
fine-tuning included. Dense embeddings are cached under `data/indexes/`; the file name
hashes the dataset, chunking and encoder settings (and the saved weights of a local
model), so changing any of them re-embeds automatically.

## Tests

```bash
python -m pytest tests/ -q
```

73 tests check the metrics against hand-worked values, the fusion, config, significance,
subset, scoring and abstention code, and that the paper's references, figures and
headline numbers still match `results/`, and that the failure review sheet matches its
worksheet. GitHub runs them on every pull request.

## Data and models

The code is MIT-licensed (below); the data and models it uses keep their own licences.
`results/` and `docs/FAILURE_REVIEW.md` quote SciFact claims and abstract excerpts.

| What | Source | Licence |
|---|---|---|
| SciFact claims and evidence annotations | Wadden et al., EMNLP 2020 ([allenai/scifact](https://github.com/allenai/scifact)), used via BEIR | CC BY 4.0 |
| SciFact abstracts | Semantic Scholar S2ORC | ODC-By 1.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face Hub | Apache 2.0 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face Hub | Apache 2.0 |
| `google/flan-t5-base` | Hugging Face Hub | Apache 2.0 |
| `deepset/roberta-base-squad2` | Hugging Face Hub | CC BY 4.0 |
| `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base` | Hugging Face Hub | MIT |

Licences as stated by each source on 9 Oct 2026.

## License

MIT — see [LICENSE](LICENSE).
