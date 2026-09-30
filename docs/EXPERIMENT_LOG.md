# Experiment log

Append an entry the moment a run finishes — including runs that did not work. Failed
experiments are paper content ("we tried X, it did not help, here is the number"), and
they are also the fastest way to stop yourself repeating a run you already did.

Copy the template. Do not edit old entries; add a new one instead.

---

## Template

```
### YYYY-MM-DD — <one-line description>
Commit:      <git rev-parse --short HEAD>
Config:      configs/scifact.yaml  (changed: <key>=<value>, else "unchanged")
Command:     <exact command line>
Hardware:    <GPU>, batch <n>

Result:      nDCG@10 = ...   Recall@10 = ...   Recall@100 = ...   MRR@10 = ...
Latency:     ... ms/query    Index build: ... s    Index size: ... MB

Observation: <what you actually noticed — one or two sentences>
Next:        <what this makes you want to run next>
```

---

## Entries

### 2026-09-28 — Baseline BM25, default k1/b
Commit:      8bec292
Config:      unchanged
Command:     `python -m hybridqa.cli ablation --variants bm25`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126 (BM25 itself runs on CPU)

Result:      nDCG@10 = 0.6562   Recall@10 = 0.7839   Recall@100 = 0.8818   MRR@10 = 0.6232
Latency:     61.6 ms/query (BM25 row of the full run below; this run wrote to the same results/
             folder, so its own files were overwritten. Accuracy re-confirmed identical on 2026-09-29.)
             Index build: BM25 22.4 s

Observation: Recall@10 is inside the expected 0.6–0.8 band, and nDCG@10 is within 0.01 of the
             BM25 figure BEIR reports for SciFact (≈0.665), so loading, chunking (21,015 passages,
             4.05 per abstract) and evaluation behave as intended.
Next:        all four variants plus the depth and RRF-k sweeps.

---

### 2026-09-28 — Full four-way ablation with rerank-depth and RRF-k sweeps
Commit:      8bec292 (results committed in a866a59)
Config:      unchanged
Command:     `python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100`
Hardware:    as above

Result:      variant         nDCG@10  R@10    R@100   MRR@10  ms/query
             bm25            0.6562   0.7839  0.8818  0.6232   61.6
             dense           0.6789   0.8149  0.9160  0.6405   84.8
             fusion          0.7097   0.8326  0.9470  0.6751  125.2
             fusion+rerank   0.7154   0.8329  0.9470  0.6850  395.5
             Rerank depth N -> nDCG@10 (ms/query): 10 -> 0.7218 (209), 25 -> 0.7177 (171),
             50 -> 0.7154 (356), 100 -> 0.7163 (497)
             RRF k -> nDCG@10: 10 -> 0.7139, 30 -> 0.7132, 60 -> 0.7097, 100 -> 0.7093
Latency:     see table. Dense embeddings were loaded from cache (dense_from_cache = 1), so
             dense_index_s = 0.1 s in metrics.json is NOT a build time; see the 2026-09-29 timing entry.

Observation: Fusion gives the clear gain over both single retrievers. Reranking adds only +0.006
             nDCG@10 over fusion at about 3x the latency, and accuracy stays flat across depths
             10-100. RRF is insensitive to k. Latency is not monotonic in depth (N=25 faster than
             N=10), so timings carry noise.
Next:        figures; significance for every pair (only fusion+rerank vs each was tested);
             dense build time with a cold cache.

---

### 2026-09-29 — Significance for every pair of variants (no new retrieval)
Commit:      "Test every pair of variants for significance; make figures print-ready" (2026-09-29)
Config:      unchanged
Command:     `python scripts/significance.py`  (reads results/runs/*.json from the run above)
Hardware:    n/a (CPU, seconds)

Result:      paired bootstrap on nDCG@10, 10,000 resamples, seed 42; better/worse/same = queries
             dense vs bm25            +0.0227  p = 0.2473    73/60/167   not significant
             fusion vs bm25           +0.0534  p < 0.0001    76/22/202   significant
             fusion vs dense          +0.0307  p = 0.0191    70/34/196   significant
             fusion+rerank vs bm25    +0.0592  p = 0.0002    80/39/181   significant
             fusion+rerank vs dense   +0.0365  p = 0.0253    76/50/174   significant
             fusion+rerank vs fusion  +0.0057  p = 0.6533    52/46/202   not significant
Latency:     n/a

Observation: The three p-values from the original run are reproduced exactly. Dense and BM25 are
             statistically tied; fusion is significantly better than both; the reranker's gain over
             fusion is not significant (it helps 52 queries and hurts 46).
Next:        dense index build time; chunking study.

---

### 2026-09-29 — Dense index build time, cold cache
Commit:      as above
Config:      unchanged
Command:     moved data/indexes/*.npy aside, then
             `python -m hybridqa.cli ablation --variants bm25 --out results/index_build_timing`
Hardware:    as above

Result:      nDCG@10 = 0.6562   Recall@10 = 0.7839   Recall@100 = 0.8818   MRR@10 = 0.6232
             (identical to 2026-09-28; rebuilt embeddings byte-identical to the cached file)
Latency:     BM25 112.2 ms/query. Index build: dense 129.6 s (21,015 passages, 32.3 MB, includes
             model load); BM25 35.5 s; load + chunk 17.3 s

Observation: Accuracy is fully deterministic, but the same work timed very differently from the
             2026-09-28 run (BM25 index 22.4 s vs 35.5 s; BM25 latency 61.6 vs 112.2 ms/query).
             Timings on this laptop can vary about 2x between runs: report them as approximate,
             single-machine numbers.
Next:        before quoting latency in the paper, consider the median of 3 timing runs.

---

### 2026-09-30 — Chunking study: whole abstracts vs 3-sentence windows
Commit:      0c13422 (plus scripts/compare_runs.py, committed 2026-09-30)
Config:      changed: chunking.strategy=none (default: sentence_window, 3 sentences, stride 2)
Command:     `python scripts/run_ablation.py --set chunking.strategy=none`
             then `python scripts/compare_runs.py "results/set_chunking.strategy=none"`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      whole abstracts (5,183 passages, 1 per document); results/set_chunking.strategy=none/
             variant         nDCG@10  R@10    MRR@10  ms/query
             bm25            0.6579   0.7829  0.6228   34.3
             dense           0.6484   0.7883  0.6068   25.5
             fusion          0.7030   0.8409  0.6674   57.3
             fusion+rerank   0.6866   0.8089  0.6554  452.7
             nDCG@10 change vs sentence windows (paired bootstrap, 10,000 resamples):
             bm25            +0.0017  p = 0.8160  not significant
             dense           -0.0305  p = 0.0090  significant
             fusion          -0.0066  p = 0.4523  not significant
             fusion+rerank   -0.0288  p = 0.0067  significant
Latency:     see table. Index build: dense 48.1 s (8.0 MB) vs 129.6 s (32.3 MB) with sentence
             windows; BM25 16.8 s. Total run 263 s.

Observation: Chunking does not matter for BM25, but it significantly helps the two neural
             components: the dense encoder and the cross-encoder both do worse on whole abstracts.
             Measured cause for the dense encoder: 71.6% of whole abstracts (median 317 tokens;
             75.3% of the gold-evidence abstracts) exceed its 256-token limit and are cut off,
             against 1.2% of the 3-sentence windows (median 125 tokens). BM25 reads every word
             either way.
             Within the whole-abstract setting fusion still beats both single retrievers
             (p = 0.0002 vs BM25, p < 0.0001 vs dense), so the fusion result does not depend on
             the chunking choice. Chunking costs 4x the index size and about 2x first-stage latency.
             Recall@100 is NOT comparable between the two settings: with sentence windows the
             top-100 passages cover only about 47-58 distinct documents per query, with whole
             abstracts exactly 100. Compare the two on @10 metrics only.
Next:        rank fusion vs score fusion (`--variants fusion --set fusion.method=nsf`).

---

### 2026-09-30 — Rank fusion (RRF) vs score fusion (min-max normalised, equal weights)
Commit:      dfe51b3
Config:      changed: fusion.method=nsf (default: rrf, k = 60)
Command:     `python scripts/run_ablation.py --variants fusion --set fusion.method=nsf`
             then `python scripts/compare_runs.py "results/set_fusion.method=nsf"`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      score fusion: nDCG@10 = 0.7174   Recall@10 = 0.8368   MRR@10 = 0.6826
             RRF (main run): nDCG@10 = 0.7097  Recall@10 = 0.8326   MRR@10 = 0.6751
             difference +0.0078, p = 0.2728, better/worse/same = 32/26/242: not significant
Latency:     235.2 ms/query (RRF in the main run: 125.2). Both do the same retrieval work, so
             this gap is timing noise, not a cost of the method. Total run 98 s.

Observation: Score fusion is numerically a little higher, but the two methods are statistically
             tied on SciFact. So the data does NOT show that rank fusion beats score fusion. RRF
             stays the default for a different reason: it needs no score normalisation and is
             insensitive to its one parameter (k = 10-100 gives 0.709-0.714), which matters when
             there is no labelled data to tune on.
Next:        paper section 3 (Method). Optional: a stronger dense encoder, reported alongside
             MiniLM, not instead of it.

---

<!-- add new entries above this line, newest last -->
