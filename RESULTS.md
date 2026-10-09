# Results: what was run, what came out, and what was done next

This page tells the project as it happened: each experiment, the command that ran it, the
output it printed, what that output meant, and the decision it led to. The numbers are
copied from the files in [`results/`](results/); the full entries (commit, config,
hardware) are in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

Unless stated otherwise, scores are **nDCG@10 on the 300 SciFact test claims** (1.0 = every
relevant abstract ranked at the top), and *p* is a paired bootstrap test (10,000 resamples):
*p* < 0.05 means the difference is unlikely to be luck.

## The path at a glance

| # | Date | Experiment | Output | Decision |
|---|---|---|---|---|
| 1 | 28 Sep | BM25 baseline | 0.656 (BEIR reports 0.665) | Pipeline works: run all four variants |
| 2 | 28 Sep | Four-way ablation | fusion 0.710, + reranker 0.715 | Test which differences are real |
| 3 | 29 Sep | Significance tests | fusion significant, reranker not | Measure cost; question the reranker |
| 4 | 30 Sep | Whole abstracts vs passages | dense −0.031 (p = 0.009) | Keep 3-sentence passages |
| 5 | 30 Sep | Score fusion vs RRF | tied (p = 0.27) | Keep RRF |
| 6 | 30 Sep | Stronger encoder (BGE-small) | dense 0.723; fusion no longer helps | The encoder matters: fine-tune one |
| 7 | 30 Sep | Stronger reranker (BGE) | 0.714, 1.3 s per claim | Reranking does not pay off here |
| 8 | 30 Sep | Fine-tune the encoder | dense 0.742 (+0.063, p < 0.0001) | Equal-weight fusion now hurts: tune its weight |
| 9 | 1 Oct | Which claims each retriever wins | word overlap decides | Explains when fusion helps |
| 10 | 1–2 Oct | The answer generator | "[3]." for 49 of 50 claims | Change the prompt, not the model |
| 11 | 2 Oct | Tuned fusion weight (0.3) | **0.759**, best (p = 0.003 vs original) | Final system |
| 12 | 2–3 Oct | Failure analysis, answer scoring, abstention | 42% missing evidence; AUC 0.81 | Limits and future work |
| 13 | 2–3 Oct | Reproduce from a fresh clone | identical, training included | Results are reproducible |

---

## 1. BM25 baseline (28 Sep)

```
python -m hybridqa.cli ablation --variants bm25
```
Output:
```
nDCG@10 = 0.6562   Recall@10 = 0.7839   Recall@100 = 0.8818   MRR@10 = 0.6232
```
**What it told us.** BEIR reports 0.665 for BM25 on SciFact, so our data loading, passage
splitting (5,183 abstracts → 21,015 passages) and evaluation behave as intended.
**Next:** run all four retrieval variants on the same index.

## 2. The four-way ablation (28 Sep)

```
python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100
```
Output (`results/ablation.csv`; latency from the re-timed run of 1 Oct, see below):
```
variant         nDCG@10  R@10    R@100   MRR@10  ms/claim
bm25            0.6562   0.7839  0.8818  0.6232    80
dense           0.6789   0.8149  0.9160  0.6405    22
fusion          0.7097   0.8326  0.9470  0.6751   119
fusion+rerank   0.7154   0.8329  0.9470  0.6850   329
rerank depth N: 10 -> 0.7218, 25 -> 0.7177, 50 -> 0.7154, 100 -> 0.7163
RRF k:          10 -> 0.7139, 30 -> 0.7132, 60 -> 0.7097, 100 -> 0.7093
```
**What it told us.** Fusing the two retrievers gives the clear gain. The reranker adds only
+0.006 at about three times the time per claim, and more reranking depth does not help.
**Next:** check which of these differences are statistically real.

## 3. Which differences are real? (29 Sep)

```
python scripts/significance.py
```
Output (`results/significance.json`; better / worse / same = claims):
```
dense vs bm25            +0.0227  p = 0.2473    73/60/167   not significant
fusion vs bm25           +0.0534  p < 0.0001    76/22/202   significant
fusion vs dense          +0.0307  p = 0.0191    70/34/196   significant
fusion+rerank vs fusion  +0.0057  p = 0.6533    52/46/202   not significant
```
**What it told us.** BM25 and dense retrieval are tied overall but win on *different* claims
(73 vs 60), which is why fusing them helps. The reranker's gain is not real: it helps 52
claims and hurts 46.
**Next:** test the design choices one at a time.

## 4. Passages or whole abstracts? (30 Sep)

```
python scripts/run_ablation.py --set chunking.strategy=none
python scripts/compare_runs.py "results/set_chunking.strategy=none"
```
Output:
```
                 whole abstracts   change vs passages
bm25             0.6579            +0.0017  p = 0.8160  not significant
dense            0.6484            -0.0305  p = 0.0090  significant
fusion+rerank    0.6866            -0.0288  p = 0.0067  significant
```
**What it told us.** The dense model reads at most 256 tokens, and 71.6% of whole abstracts
are longer, so they get cut off (only 1.2% of passages are). **Decision:** keep the
3-sentence passages.

## 5. Rank fusion or score fusion? (30 Sep)

```
python scripts/run_ablation.py --variants fusion --set fusion.method=nsf
```
Output: score fusion 0.7174 vs RRF 0.7097, difference +0.0078, **p = 0.2728** (not significant).
**Decision:** keep RRF: it is tied on accuracy and needs no score normalisation or tuning.

## 6. A stronger encoder (30 Sep)

```
python scripts/run_ablation.py --set dense.model=BAAI/bge-small-en-v1.5 --set "dense.query_prefix=Represent this sentence for searching relevant passages: " --out results/dense_bge_small
```
Output:
```
dense    0.7232   +0.0442 vs MiniLM  p = 0.0021  significant
fusion   0.7214   no better than dense alone (p = 0.87)
```
**What it told us.** With one retriever clearly stronger, equal-weight fusion adds nothing:
fusion helps only when the two retrievers are of similar strength.
**Next:** try a stronger reranker, and adapt the small encoder to scientific text.

## 7. A stronger reranker (30 Sep)

```
python scripts/run_ablation.py --variants fusion+rerank --set rerank.model=BAAI/bge-reranker-base ... --out results/dense_bge_small__rerank_bge_base
```
Output: 0.7144 (vs 0.7123 with the small reranker, p = 0.86) at **1,308 ms per claim**.
**What it told us.** A reranker eight times larger gives the same accuracy: reranking does not
pay off on SciFact in this pipeline. (This run also exposed a timing problem; see "Problems
found along the way" below.)

## 8. Fine-tuning the encoder (30 Sep)

```
python scripts/finetune_dense.py --epochs 12
python scripts/run_ablation.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact --out results/dense_minilm_finetuned
```
Training output (120 held-out development claims, never the test claims):
```
dev nDCG@10 by epoch 0-12: 0.691 0.703 0.730 0.730 0.735 0.751 0.749 0.748 0.758 0.767 0.768 0.767 0.770
```
Test output:
```
dense            0.7416   +0.0626 vs before fine-tuning  p < 0.0001  significant
fusion           0.7313   below dense alone (-0.0102, p = 0.44)
```
**What it told us.** Fine-tuning gives the largest gain of anything tried, at no extra cost
per claim. But equal-weight fusion now *lowers* the score: BM25's vote is too strong.
**Next:** find out on which claims each retriever wins, and tune the fusion weight.

## 9. Which retriever wins which claims? (1 Oct)

```
python scripts/query_subsets.py
```
Output (`results/query_subsets.json`):
```
group                          claims  bm25   dense  fusion
low word overlap (paraphrased)  156    0.441  0.515  0.526   dense beats BM25 +0.073 (p = 0.022)
high word overlap               144    0.889  0.857  0.909   fusion beats dense +0.052 (p < 0.0001)
```
**What it told us.** How many words a claim shares with its evidence decides which retriever
wins. This is one of the project's main findings.

## 10. The answer generator (1–2 Oct)

```
python -m hybridqa.cli answer --limit 50
```
Output: flan-t5-base answered **"[3]."** (a citation and nothing else) for **49 of 50 claims**.
A three times larger model (flan-t5-large) still did so for 47 of 50.
**What it told us.** The problem was the instruction, not the model size. A new prompt
("explain in one sentence whether the passages support the claim", citation added
afterwards), chosen on 20 *training* claims, gave real answers for **50 of 50** test claims.

## 11. The tuned fusion weight: the final system (2 Oct)

```
python scripts/tune_fusion.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact
python scripts/run_ablation.py --variants fusion fusion+rerank --set dense.model=data/models/all-MiniLM-L6-v2-scifact --set "fusion.weights=[0.3, 1.0]" --out results/dense_minilm_finetuned__fusion_weighted
```
Output (weight chosen on the development claims; test claims read once):
```
dev:  BM25 weight 0 -> 0.770, 0.3 -> 0.788, 1.0 -> 0.772        chosen: 0.3
test: fusion  nDCG@10 0.7587  Recall@10 0.905  114 ms per claim
      vs the original pipeline (0.7154): +0.0433  p = 0.0027  significant
```
**What it told us.** The best system: the fine-tuned encoder fused with BM25 at weight 0.3,
better than the original pipeline at about a third of its time per claim. Analysing the
fusion formula showed why: below a weight of 61/160 ≈ 0.38, BM25 can only re-order the dense
model's candidates, never add new ones.

## 12. What still goes wrong (2–3 Oct)

- **Failure analysis** (`results/failure_summary.json`): of the 40 hardest claims, 17 (42%) have
  a marked evidence abstract that does not actually state the claim, 11 (28%) are vocabulary
  mismatches, 8 (20%) are reranker errors and 4 (10%) are ambiguous claims.
- **Answer quality** (`results/answer_quality.json`, 0–2 scale): the generator is right more
  often than a copy-a-sentence baseline (0.84 vs 0.48) but less faithful (1.16 vs 2.00).
- **Abstention** (`python scripts/abstention.py`): the reranker's score predicts missing
  evidence with AUC 0.81, but a threshold that catches 68 of 74 misses also refuses 109 of
  226 answerable claims.

The failure labels and answer scores were assigned with AI assistance following fixed
definitions (`docs/ANSWER_RUBRIC.md`); see the paper and the experiment log.

## 13. Reproducing everything from a fresh clone (2–3 Oct)

A fresh clone from GitHub, with nothing cached, reproduced all the retrieval numbers exactly
and, on 3 Oct, the whole best system *including the fine-tuning*: every epoch's development
score, the chosen weight (0.3) and all 21 test metrics were identical.

---

## Problems found along the way, and how they were fixed

| When | Problem | How it was found | Fix |
|---|---|---|---|
| 25 Sep | The reranker's output was cut to 10 passages before evaluation, making its Recall@100 incomparable | End-to-end check before the first reported run | Keep the full reranked list |
| 28 Sep | Changing the passage settings could reuse stale embeddings from the cache | Code review of the cache | Cache file named by a hash of every relevant setting |
| 30 Sep | Latency included loading the model into the first timed claim (the BGE reranker run recorded 3.9 s per claim) | The same run timed very differently twice | One untimed warm-up claim before timing; main run re-timed on 1 Oct: dense 85 → 22 ms |
| 1 Oct | The answer generator replied with a citation only | Reading its 50 answers | New prompt chosen on training claims (step 10) |

## Where to find the outputs

| File | What it holds |
|---|---|
| `results/ablation.csv`, `results/metrics.json` | The four-way ablation |
| `results/significance.json` | Every pairwise test |
| `results/set_*/`, `results/dense_*/` | Each one-change configuration |
| `results/finetune_all-MiniLM-L6-v2-scifact/history.json` | Fine-tuning, epoch by epoch |
| `results/query_subsets.json` | Which retriever wins which claims |
| `results/failure_analysis.csv`, `results/failure_summary.json` | The failure analysis |
| `results/answer_quality.json`, `results/answers.json` | Answer quality and the answers |
| `results/abstention.json` | The abstention check |
