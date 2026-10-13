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

### 2026-09-30 — Stronger dense encoder: BAAI/bge-small-en-v1.5 (reported next to MiniLM)
Commit:      d651ae6 (plus the literal-string `--set` fix, committed 2026-09-30)
Config:      changed: dense.model=BAAI/bge-small-en-v1.5 and its required query prefix
             dense.query_prefix="Represent this sentence for searching relevant passages: "
             (one logical change: the encoder). Everything else unchanged, incl. max_seq_length 256.
Command:     `python scripts/run_ablation.py --set dense.model=BAAI/bge-small-en-v1.5 --set "dense.query_prefix=Represent this sentence for searching relevant passages: " --out results/dense_bge_small`
             then `python scripts/compare_runs.py results/dense_bge_small`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      variant         nDCG@10  R@10    MRR@10  ms/query   vs MiniLM run (nDCG@10)
             bm25            0.6562   0.7839  0.6232  118.9      identical (no dense model involved)
             dense           0.7232   0.8523  0.6866   32.5      +0.0442  p = 0.0021  significant
             fusion          0.7214   0.8507  0.6843  135.0      +0.0117  p = 0.1819  not significant
             fusion+rerank   0.7123   0.8274  0.6833  398.2      -0.0031  p = 0.4316  not significant
             Within this run: dense vs bm25 +0.0669 (p = 0.0003); fusion vs dense -0.0018
             (p = 0.87); fusion+rerank vs fusion -0.0092 (p = 0.50).
Latency:     see table (BM25 at 119 ms here vs 62 ms in the main run: timing noise).
             Index build: dense 240.9 s (12-layer model, 32.3 MB); model download 129 MB to D:.

Observation: The stronger encoder lifts dense retrieval significantly and makes it clearly
             better than BM25. But fusion no longer helps: it equals dense alone. With MiniLM the
             two retrievers were tied and fusion beat both; with BGE one retriever dominates and
             equal-weight RRF adds nothing. So the fusion gain depends on the two retrievers
             being of similar strength. The MS MARCO MiniLM reranker still adds nothing.
Next:        a stronger reranker on this setup (BAAI/bge-reranker-base).

---

### 2026-09-30 — Stronger reranker: BAAI/bge-reranker-base (on the BGE dense setup)
Commit:      d651ae6 (plus the warm-up fix in pipeline.py, committed 2026-09-30)
Config:      changed vs the previous entry: rerank.model=BAAI/bge-reranker-base
             (dense stays BAAI/bge-small-en-v1.5 with its query prefix; depth 50 unchanged)
Command:     `python scripts/run_ablation.py --variants fusion+rerank --set dense.model=BAAI/bge-small-en-v1.5 --set "dense.query_prefix=Represent this sentence for searching relevant passages: " --set rerank.model=BAAI/bge-reranker-base --out results/dense_bge_small__rerank_bge_base`
             then `python scripts/compare_runs.py results/dense_bge_small__rerank_bge_base --baseline results/dense_bge_small`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126; 3.9 of 4 GB used at batch 64

Result:      fusion+rerank: nDCG@10 = 0.7144   Recall@10 = 0.8438   MRR@10 = 0.6807
             vs the MiniLM reranker on the same candidates (0.7123): +0.0022, p = 0.8644, 48/52/200
             vs BGE dense alone (0.7232): -0.0087, p = 0.5058
             vs BGE fusion (0.7214): -0.0070, p = 0.5774
             vs BM25 (0.6562): +0.0582, p = 0.0001
Latency:     1307.5 ms/query (MiniLM reranker on this setup: 398 ms; fusion alone: 135 ms).
             Model download 1.1 GB to D:. The run was done twice: the first recorded 3932 ms/query
             because the model download happened inside the first timed query. run_variant now
             runs one untimed warm-up query first; accuracy was identical in both runs.

Observation: A cross-encoder about 8x heavier gives the same accuracy as the small one and no
             gain over first-stage retrieval, at about 10x the latency of fusion. Two different
             rerankers agree, so "reranking does not pay off on SciFact claims in this pipeline"
             is not a quirk of one model.
             Latencies of runs before this fix include model loading in their first query (a few
             seconds spread over 300 queries): re-time the main run before quoting final numbers.
Next:        choose what the paper reports: MiniLM as the controlled baseline, BGE as the
             stronger-encoder check.

---

### 2026-09-30 — Fine-tuning the dense encoder on SciFact train claims (training run)
Commit:      9d89cb5 plus scripts/finetune_dense.py (committed 2026-09-30)
Config:      base model sentence-transformers/all-MiniLM-L6-v2; contrastive loss (multiple-negatives
             ranking, in-batch negatives, scale 20); AdamW lr 2e-5, weight decay 0.01, 10% warm-up
             then linear decay; batch 32; max_seq_length 256; seed 42
Command:     `python scripts/finetune_dense.py --epochs 12`
             (first `python scripts/finetune_dense.py`, 6 epochs: dev was still rising, see below)
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      data: 809 train claims (no overlap with the 300 test claims) -> 689 for training
             (792 claim-passage pairs), 120 held out as dev. Test split never loaded.
             6-epoch trial, dev nDCG@10 by epoch 0-6: 0.691 0.706 0.733 0.733 0.738 0.742 0.745
             12-epoch run, dev nDCG@10 by epoch 0-12:
               0.691 0.703 0.730 0.730 0.735 0.751 0.749 0.748 0.758 0.767 0.768 0.767 0.770
             training loss by epoch 1-12:
               0.264 0.162 0.117 0.093 0.087 0.080 0.066 0.052 0.073 0.070 0.068 0.080
             kept: epoch 12 (best dev nDCG@10 = 0.7704; dev Recall@10 0.829 -> 0.888)
Latency:     n/a. Training took 584 s including a full-corpus dev evaluation after every epoch.
             Model saved to data/models/all-MiniLM-L6-v2-scifact (88 MB, not committed).

Observation: The loss falls and dev accuracy rises from 0.691 to 0.770, flattening from epoch 9
             (0.767-0.770), so 12 epochs is enough. The training length was decided on the dev
             set only, before the test set was evaluated. The loss is not perfectly monotonic
             after epoch 8: with 24 batches per epoch and random in-batch negatives it is noisy.
Next:        evaluate once on the test claims.

---

### 2026-09-30 — Fine-tuned dense encoder on the TEST claims
Commit:      as above
Config:      changed: dense.model=data/models/all-MiniLM-L6-v2-scifact (everything else unchanged)
Command:     `python scripts/run_ablation.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact --out results/dense_minilm_finetuned`
             then `python scripts/compare_runs.py results/dense_minilm_finetuned`
Hardware:    as above

Result:      variant         nDCG@10  R@10    MRR@10  ms/query   vs the same model before fine-tuning
             bm25            0.6562   0.7839  0.6232   80.0      identical
             dense           0.7416   0.8860  0.6994   15.8      +0.0626  p < 0.0001  significant
             fusion          0.7313   0.8875  0.6875   93.1      +0.0216  p = 0.0129  significant
             fusion+rerank   0.7245   0.8484  0.6925  255.5      +0.0092  p = 0.1064  not significant
             Within this run: dense vs bm25 +0.0853 (p < 0.0001); fusion vs dense -0.0102 (p = 0.44);
             fusion+rerank vs fusion -0.0068 (p = 0.64).
             Fine-tuned dense vs the original pipeline: vs fusion +0.0319 (p = 0.043); vs
             fusion+rerank +0.0262 (p = 0.13, not significant).
             Fine-tuned MiniLM dense vs off-the-shelf BGE-small dense (0.7232): +0.0184, p = 0.23.
             Claims with a gold paper in the top 10: 269 of 300 (MiniLM before: 248; BM25: 240).
Latency:     see table (with the warm-up fix; not comparable to the earlier main-run timings).
             Index build: dense 53.3 s (32.3 MB), same size and speed as the base model.

Observation: In-domain fine-tuning gives the largest gain of any change tried: dense retrieval
             +0.063 nDCG@10 on unseen test claims, at no extra query cost (same model size).
             The dev gain (+0.079) carries over to test (+0.063), so it is not overfitting to dev.
             A 6-layer model fine-tuned on 689 claims is statistically level with the larger
             off-the-shelf BGE-small. As with BGE, once dense is clearly stronger than BM25,
             equal-weight fusion no longer adds anything, and the reranker still does not help.
             Honest limit: against the original full pipeline (0.715) the gain is not significant.
Next:        paper: method (fine-tuning subsection) and results; failure analysis on this setup.

---

### 2026-10-01 — Query-subset analysis: which retriever helps which kind of claim
Commit:      7e84c34
Config:      unchanged (analysis of saved runs; no retrieval)
Command:     `python scripts/query_subsets.py` and
             `python scripts/query_subsets.py --results results/dense_minilm_finetuned`
Hardware:    n/a (CPU, seconds)

Result:      nDCG@10 per group, main configuration (pretrained MiniLM):
             group                          n   bm25   dense  fusion  dense-bm25 (p)   fusion-dense (p)
             low word overlap (paraphrased) 156  0.441  0.515  0.526  +0.073 (0.022)   +0.011 (0.60)
             high word overlap              144  0.889  0.857  0.909  -0.032 (0.13)    +0.052 (<0.0001)
             contains a number              114  0.655  0.715  0.703  +0.060 (0.031)   -0.011 (0.59)
             no number                      186  0.657  0.657  0.714  +0.000 (1.00)    +0.056 (0.0005)
             rare term (rarest df <= 5)     163  0.721  0.721  0.757  -0.000 (1.00)    +0.035 (0.032)
             common terms only              137  0.579  0.629  0.654  +0.050 (0.093)   +0.025 (0.22)
             Fine-tuned MiniLM: dense beats BM25 in every group except high overlap
             (-0.016, p = 0.42). Fusion vs dense: high overlap +0.040 (p = 0.0006), low overlap
             -0.057 (p = 0.014). Overlap split at the median, 0.571 of the claim's terms.
Latency:     n/a

Observation: Word overlap with the evidence is what separates the retrievers. On paraphrased
             claims the dense retriever is significantly better than BM25; on high-overlap claims
             BM25 is at least as good, and that is where fusion gains over dense. With the
             fine-tuned encoder fusion helps on high-overlap claims and hurts on paraphrased ones,
             and the two effects cancel, which is why fusion adds nothing on average there. This
             suggests weighting the retrievers per claim (future work).
             The expected advantage of BM25 on claims containing numbers or codes is NOT found:
             dense is better on them (+0.060, p = 0.031). Low-overlap claims are much harder for
             every method (best 0.53 vs 0.92).
             Caveat: many groups and tests (no multiple-comparison correction); "overlap" uses the
             gold abstract, so it describes the claims and cannot be used by a retriever.
Next:        figure for the overlap split; paper subsection "Which stage helps which question?".

---

### 2026-10-01 — Main ablation re-timed with the warm-up query
Commit:      d8d9b67 (pipeline with the untimed warm-up query, added 2026-09-30)
Config:      unchanged
Command:     `python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      every accuracy number and every p-value is identical to the 2026-09-28 run
             (deterministic retrieval); only latency changed:
             variant         ms/query before -> after the warm-up fix
             bm25              61.6 ->  79.8
             dense             84.8 ->  21.9
             fusion           125.2 -> 119.3
             fusion+rerank    395.4 -> 329.0
             rerank depth N=10/25/50/100: 190 / 226 / 331 / 468 ms (before: 209 / 171 / 356 / 497)
Latency:     see above. Total run 830 s.

Observation: The old dense latency included loading the encoder into the first query: the
             real cost is about 22 ms, a quarter of what the first run recorded. The depth sweep
             is now monotonic (the old run had N=25 faster than N=10). BM25 varies between runs
             (62-120 ms seen so far): it scores every passage in pure Python. These numbers
             replace the old ones in the paper and README.
Next:        update the ms/q numbers in the paper, README and viva prep.

---

### 2026-10-01 — 50 answers from each reader, and the blind scoring sheet
Commit:      48cf649
Config:      main configuration; answer.mode=generative (default) and answer.mode=extractive
Command:     `python -m hybridqa.cli answer --limit 50`
             `python -m hybridqa.cli answer --limit 50 --set answer.mode=extractive`
             `python scripts/answer_sheet.py`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      both readers get the same 5 passages (fusion + reranking) for the first 50 test
             claims; the gold abstract is among them for 41 of the 50 claims (82 of 100 sheet rows).
             automatic citation metrics vs the gold abstracts:
               generative (flan-t5-base):        precision 0.60, recall 0.56
               extractive (roberta-base-squad2): precision 0.70, recall 0.67
             generative answers that are only a citation (e.g. "[3]."): 49 of 50;
             "Not enough evidence": 0 for either reader.
Latency:     not measured.

Observation: flan-t5-base with this prompt almost never writes an answer, only a passage
             number, so as configured the generative reader adds nothing but a citation; the
             extractive reader always returns a span from the top passage. The automatic citation
             numbers mostly reflect which passage is cited, not answer quality, which is why the
             answers are scored by hand. Because the outputs look so different, the scoring can
             only be partly blind (rubric updated).
Next:        hand-score the sheet (docs/ANSWER_RUBRIC.md). Decision for the reader: keep and
             report, or try google/flan-t5-large (RUN_AND_PRESENT §5) before scoring.

---

### 2026-10-02 — Document-level fusion
Commit:      676612d
Config:      changed: fusion.level=document (default: passage); variants fusion and fusion+rerank
Command:     `python scripts/run_ablation.py --variants fusion fusion+rerank --set fusion.level=document`
             `python scripts/run_ablation.py --variants fusion fusion+rerank --set dense.model=data/models/all-MiniLM-L6-v2-scifact --set fusion.level=document --out results/dense_minilm_finetuned__doc_fusion`
             then `scripts/compare_runs.py` against the passage-level runs
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      nDCG@10, passage-level -> document-level fusion (paired bootstrap):
             pretrained MiniLM   fusion          0.7097 -> 0.7062  -0.0034  p = 0.46   17/31/252
                                 fusion+rerank   0.7154 -> 0.7041  -0.0113  p = 0.11   21/38/241
             fine-tuned MiniLM   fusion          0.7313 -> 0.7228  -0.0085  p = 0.12   18/42/240
                                 fusion+rerank   0.7245 -> 0.7086  -0.0160  p = 0.0195 19/39/242
Latency:     not compared (a 3 GB model download ran at the same time). Runs took 156-191 s.

Observation: Fusing documents instead of passages does not help, and with the fine-tuned encoder
             plus reranking it is significantly worse. It fixes cases like claim 1259, but those
             are rare. Likely reason: at passage level a paper has several chances (whichever of
             its passages ranks highest counts), while at document level only one passage per paper
             reaches the reranker, and if it is not the one with the evidence the paper is lost.
             Passage-level fusion stays the default; the 1259 case remains a documented limitation.
Next:        flan-t5-large reader.

---

### 2026-10-02 — A larger reader: flan-t5-large (bfloat16)
Commit:      5e4e1aa (answer.dtype)
Config:      changed: answer.generative_model=google/flan-t5-large, answer.dtype=bfloat16
Command:     `python -m hybridqa.cli answer --limit 50 --set answer.generative_model=google/flan-t5-large --set answer.dtype=bfloat16 --out results/answers_flan_t5_large`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB); model download 3.1 GB to data/hf_cache (D:)

Result:      first 50 test claims, same passages as before
             reader                    citation-only answers   automatic citation precision / recall
             flan-t5-base  (float32)        49 / 50             0.60 / 0.56
             flan-t5-large (bfloat16)       47 / 50             0.70 / 0.65
             extractive (roberta-squad2)     0 / 50             0.70 / 0.67
Latency:     not measured.

Observation: Three times the parameters do not fix the reader: it still answers with a passage
             number. The cause is the task framing, not the model size: SciFact queries are claims,
             not questions, and the prompt ("Answer the question ... cite every claim with [2]")
             invites the model to pick a passage like a multiple-choice answer. The larger model
             does pick a gold passage more often (precision 0.70 vs 0.60).
Next:        rewrite the prompt for claims, choosing it on TRAINING claims, not test claims.

---

### 2026-10-02 — Reader prompt chosen on training claims; explain-then-cite on test claims
Commit:      32fb6ce
Config:      changed: answer.prompt_style=explain_then_cite (new default); flan-t5-base, float32
Command:     prompt development: a scratch script on 20 TRAINING claims (random seed 7), with
             fusion + reranking passages, flan-t5-base and flan-t5-large (bfloat16);
             then `python -m hybridqa.cli answer --limit 50` and `python scripts/answer_sheet.py`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      training claims with a real answer (>= 4 words besides citations), out of 20:
             prompt                                         base    large
             current: answer + cite inline                    0       1
             "is the claim supported or refuted?"             1       2    (one word)
             "what do the passages report about the claim?"  20      16    (mostly titles)
             "explain in one sentence whether they support"  19       0    (large: "Yes"/"No")
             chosen: "explain" with flan-t5-base, citation attached afterwards to the passage
             sharing the most terms with the sentence.
             test claims (first 50): real answers 50/50 (was 1/50); cited passage position
             1: 27, 2: 5, 3: 3, 4: 5, 5: 10; automatic citation precision 0.60, recall 0.56.
             The old citation-only answers are kept in results/answers_cite_inline/.
Latency:     not measured.

Observation: The reader's failure was the prompt, not the model: asked to explain, flan-t5-base
             writes a sentence for every claim. Quality varies by eye (some sentences restate
             the claim or miss), which the hand scoring will measure. The prompt was chosen on
             training claims only. The scoring sheet was rebuilt (it had no scores yet): both
             readers now return text and the trailing [n] is dropped, so blinding is better,
             though still partial.
Next:        hand-score results/answer_scoring.csv with docs/ANSWER_RUBRIC.md.

---

### 2026-10-02 — Reproducibility check from a fresh clone
Commit:      23f64aa (cloned from GitHub into a separate folder, _repro_check, on D:)
Config:      unchanged
Command:     `git clone https://github.com/Vivansh-07/hybrid-retrieval-qa.git _repro_check`, then in
             the clone `python -m pytest tests/ -q` and
             `python scripts/run_ablation.py --variants bm25 dense fusion --out results/_repro`
             (empty data/ folder: SciFact and MiniLM downloaded fresh, dense index rebuilt)
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      101 tracked files; the PDF, PNG and DOCX files are byte-identical to the originals;
             43 tests pass; all 21 accuracy columns of bm25, dense and fusion are identical to
             results/ablation.csv (nDCG@10 0.6562 / 0.6789 / 0.7097).
Latency:     n/a. Dense index built from scratch in 125.4 s; run took 248 s; 131 MB downloaded
             into the clone's own data/ folder on D: (nothing written to C:).

Observation: The results are reproducible from the public repository alone, with nothing cached.
             The clone was deleted afterwards.
Next:        —

---

### 2026-10-02 — Weighted fusion: BM25 weight chosen on dev claims, evaluated once on test
Commit:      bcec2c7
Config:      changed: fusion.weights=[0.3, 1.0] (BM25, dense), chosen by scripts/tune_fusion.py on
             the 120 dev claims of the training split (grid 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2)
Command:     `python scripts/tune_fusion.py` and `... --set dense.model=data/models/all-MiniLM-L6-v2-scifact`
             `python scripts/run_ablation.py --variants fusion fusion+rerank --set "fusion.weights=[0.3, 1.0]" --out results/fusion_weighted`
             `python scripts/run_ablation.py --variants fusion fusion+rerank --set dense.model=data/models/all-MiniLM-L6-v2-scifact --set "fusion.weights=[0.3, 1.0]" --out results/dense_minilm_finetuned__fusion_weighted`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), PyTorch 2.6.0+cu126

Result:      dev (training split, 120 claims), nDCG@10 by BM25 weight:
               pretrained  0: 0.691  0.3: 0.732  1.0: 0.720     fine-tuned  0: 0.770  0.3: 0.788  1.0: 0.772
             test (300 claims), evaluated once:
             setup                         variant         nDCG@10  R@10   MRR@10  ms/q
             fine-tuned, weighted fusion   fusion          0.7587   0.905  0.718   114
                                           fusion+rerank   0.7445   0.883  0.708   306
             pretrained, weighted fusion   fusion          0.7109   0.839  0.674   117
                                           fusion+rerank   0.7286   0.857  0.696   315
             paired bootstrap, nDCG@10:
               fine-tuned: weighted vs equal-weight fusion   +0.0273  p = 0.0009
               fine-tuned: weighted fusion vs dense alone    +0.0171  p = 0.057
               fine-tuned: weighted vs equal, + rerank       +0.0199  p = 0.0001
               pretrained: weighted vs equal-weight fusion   +0.0013  p = 0.88
               pretrained: weighted vs equal, + rerank       +0.0132  p = 0.051
               fine-tuned weighted fusion vs ORIGINAL full pipeline (0.7154): +0.0433  p = 0.0027
Latency:     see table.

Observation: Best system of the project: the fine-tuned encoder fused with BM25 at weight 0.3,
             nDCG@10 0.759 and a relevant abstract in the top 10 for 90.5% of claims. Equal-weight
             fusion hurt the strong fine-tuned retriever on paraphrased claims; a smaller BM25 weight,
             learned on held-out training claims, turns fusion from a loss into a gain, and the
             improvement over the original pipeline is now significant (p = 0.003; fine-tuned dense
             alone was not, p = 0.13). With the pretrained encoder the weight barely matters, as
             expected when the two retrievers are equally strong. The reranker still does not beat
             fusion. The weight was chosen on dev claims only and the test set was read once.
Next:        report it in the paper, README and viva prep.

---

### 2026-10-02 — Failure analysis of the 40 hardest claims (AI-assisted labelling)
Commit:      1cece24
Config:      main configuration (pretrained MiniLM, equal-weight RRF, MS MARCO reranker)
Command:     `python scripts/failure_analysis.py --top-n 40` (worksheet, 1 Oct), labels written
             into results/failure_analysis.csv, then `python scripts/failure_analysis.py --summarize`
Hardware:    n/a

Result:      missing_evidence 17 (42%), terminology_mismatch 11 (28%), reranker_error 8 (20%),
             ambiguous_query 4 (10%), conflicting_evidence 0, granularity 0.
             8 of the 10 rows flagged reranker_regression were judged reranker errors.
             Median word overlap of the terminology_mismatch claims: 0.25.
Latency:     n/a

Observation: Labels were assigned by the AI assistant (Claude) from the worksheet and the full
             gold abstracts, using the fixed category definitions; Vivansh asked for this. They
             have not been reviewed by a person: review them before the viva, and say who labelled
             them in the paper (done in the Failure Analysis section).
             The largest group is not a retrieval error: the gold abstract does not state the
             claim. Three contradictory claim pairs (820/821, 830/831, 913/914) share one gold
             abstract each, and claim 437 (MDS) is paired with a glioma abstract. Claim 690 expands
             "SFM" (most likely severe falciparum malaria) as an unrelated syndrome.
Next:        answer scoring.

---

### 2026-10-02 — Answer scoring of 100 answers (AI-assisted)
Commit:      21016aa
Config:      main configuration; generative (flan-t5-base, explain_then_cite) and extractive
             (roberta-base-squad2) readers on the first 50 test claims
Command:     scores written into results/answer_scoring.csv (docs/ANSWER_RUBRIC.md), then
             `python scripts/answer_sheet.py --summarize`
Hardware:    n/a

Result:      mean 0-2 scores        generative   extractive
             correctness               0.84         0.48
             completeness              0.76         0.36
             faithfulness              1.16         2.00
             citation accuracy         1.40         1.64
             fully correct (2)         17/50        10/50
             unsupported content (F=0) 17/50         0/50
             No reader abstained on the 9 claims whose gold abstract was not among the passages.
Latency:     n/a

Observation: Scored by the AI assistant (Claude), at Vivansh's request; not blind (the assistant
             generated the answers) and no second annotator. Conventions are listed in the rubric.
             Clear trade-off: the generative reader answers more often and more usefully but
             invents content in a third of its answers (CXCR2 vs CX3CR1, FUS vs ND3/ND6, albendazole
             as a myeloma drug); the extractive reader is always faithful but mostly returns
             fragments. Neither knows when to abstain.
Next:        paper answer-quality subsection (done in the same commit series).

---

### 2026-10-03 — What the BM25 weight of 0.3 does; demo check of the best system
Commit:      f1eba53 (analysis), 4779497 (demo claims)
Config:      best system: fine-tuned MiniLM, RRF k = 60, weights BM25 / dense = 0.3 / 1.0,
             fusion.top_k = 100 passages
Command:     saved runs in results/dense_minilm_finetuned/ and
             results/dense_minilm_finetuned__fusion_weighted/ (a scratch script, no retrieval);
             `python scripts/pick_demo_claims.py`; `python scripts/demo.py --best`
Hardware:    RTX 3050 Laptop GPU (demo only)

Result:      No document in the weighted fusion lists lies outside the dense retriever's list
             (0 over 300 claims). From the formula: a BM25-only passage scores at most
             0.3/61 = 0.0049 < 1/160 = 0.0063 for dense's 100th passage, and fusion keeps 100,
             so BM25 only re-orders dense's candidates. Break-even weight 61/160 = 0.38.
             Claims with gold in BM25's top 10 but absent from the dense list: 2 (303, 660);
             equal-weight fusion keeps 1 of them in its top 10, weighted fusion neither.
             Demo (--best): claims 269, 1303, 659 and 660 show the same top-5 ranks as the runs.
Latency:     demo startup 2 min (BM25 index 28 s, model warm-up 82 s); per claim 17-47 ms for
             BM25/dense, 64-100 ms fusion, 0.4-0.5 s with the reranker

Observation: The 0.759 comes from re-ordering, at the cost of claims only BM25 finds. With the
             fine-tuned encoder that cost is 2 claims; with a weaker encoder it would be larger,
             so the weight should be re-tuned (or kept above 0.38) if the encoder changes.
             For claim 660 the reader, given no relevant passage, wrote "Ivermectin is a
             sedative that is used to treat onchocerciasis" citing an antipsychotics paper.
Next:        none planned; added to the paper (Results, Limitations) and viva prep.

---

### 2026-10-03 — Reproducibility of the best system from a fresh clone
Commit:      72ba527 (cloned from GitHub into a separate folder, _repro_best, on D:)
Config:      fine-tuned MiniLM, BM25 weight from tune_fusion.py, otherwise unchanged
Command:     in the clone: `python -m pytest tests -q`;
             `python scripts/finetune_dense.py --epochs 12`;
             `python scripts/tune_fusion.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact`;
             `python scripts/run_ablation.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact
             --set "fusion.weights=[0.3, 1.0]" --variants dense fusion --out results/_repro_best`
             (HF_HOME pointed at the existing download cache; model, index and results new)
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)

Result:      Fine-tuning: all 12 epochs' train loss and dev nDCG@10/Recall@10 identical to
             results/finetune_all-MiniLM-L6-v2-scifact/history.json (best epoch 12, 0.7704).
             Weight grid on the 120 dev claims identical; 0.3 chosen again (dev 0.7879).
             Test: all 21 metric columns of dense (0.7416) and weighted fusion (0.7587,
             Recall@10 0.905) identical to the committed results.
Latency:     fine-tuning 11 min 23 s, weight tuning 2 min 47 s, evaluation 1 min 19 s

Observation: The best system, training included, is exactly reproducible from the repository
             (seeded, and deterministic on this GPU). The earlier fresh-clone check (2 Oct)
             covered the main ablation only. The clone was deleted afterwards.
Next:        —

---

### 2026-10-03 — Could a retrieval signal tell the reader to abstain?
Commit:      see the commit adding scripts/abstention.py
Config:      main configuration, and the best system (`--best`)
Command:     `python scripts/abstention.py`; `python scripts/abstention.py --best`
Hardware:    RTX 3050 Laptop GPU (4 GB)

Result:      Label: is any gold abstract among the 5 passages given to the reader?
             main: yes for 226/300 test claims (74 misses); best: 235/300 (65 misses).
             ROC AUC on test          main    best
               top cross-encoder     0.807   0.782
               top dense cosine      0.726   0.718
             Threshold chosen on the 120 dev claims (maximising caught misses minus refused
             hits), applied once to test, main configuration:
               cross-encoder: abstains on 68/74 misses and 109/226 answerable claims
               dense:         abstains on 35/74 misses and  36/226 answerable claims
             Best system, cross-encoder: 59/65 misses, 118/235 answerable.
Latency:     n/a (signals come from retrieval that runs anyway)

Observation: The reranker's confidence carries real signal (AUC about 0.8), but a single
             threshold cannot buy abstention cheaply: catching most misses refuses about half
             of the answerable claims. Only 27 dev claims are misses, so the threshold itself
             is noisy. A reader that abstains needs more than a score cut-off, e.g. a trained
             verifier over the claim and passages.
Next:        one sentence in the paper (Answer quality) and a viva answer.

---

### 2026-10-09/10 — Clean local deployment following docs/DEPLOY.md
Commit:      7c8ca4c (cloned from GitHub into a new folder, D:\Projects\hybrid-retrieval-qa);
             pulled to 41eed51 before the live re-run
Config:      unchanged; models built from scratch (DEPLOY.md step 5, option B)
Command:     DEPLOY.md steps 1-7, ending with
             `python scripts/run_ablation.py --variants bm25 dense fusion --out results/_live`
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Windows 11, Python 3.13

Result:      Install: PyTorch (CUDA) 32 min, other packages 22 min; environment 5.0 GB.
             Fine-tuning: dev nDCG@10 per epoch identical to the committed history
             (0.6912 before training, 0.7704 at epoch 12); weight 0.3 chosen again (dev 0.7879).
             Live re-run (10 Oct): all 63 metric values of bm25, dense and fusion identical
             to results/ablation.csv; significance identical (fusion vs bm25 +0.0534,
             p < 0.0001; fusion vs dense +0.0307, p = 0.0191; dense vs bm25 +0.0227, p = 0.2473).
Latency:     bm25 109.7 ms/query (committed 79.8), dense 21.9 (21.9), fusion 122.8 (119.3);
             live re-run 4 min 7 s; demo first start 11 min, later about 2 min, also offline

Observation: The deployment guide works as written from a fresh clone, and every number
             except timing reproduces. Timing varies between runs, which is why the paper
             calls its ms/query figures approximate.
Next:        RESULTS.md step 14; on the day, follow docs/DEPLOY.md and docs/RUN_AND_PRESENT.md.

---

### 2026-10-10 — Chunker bug: sentences after a short window were never indexed
Commit:      be840e4 (branch fix/chunker-keep-tail; issue #45)
Config:      changed: chunking.keep_tail=true (default false = every reported result)
Command:     `python scripts/run_ablation.py --set chunking.keep_tail=true`;
             `python scripts/run_ablation.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact
             --set "fusion.weights=[0.3, 1.0]" --set chunking.keep_tail=true --variants dense fusion
             --out results/dense_minilm_finetuned__fusion_weighted__keep_tail`;
             `python scripts/compare_runs.py` on each folder
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)

Result:      Found by a code review: chunk_document stopped at the first window shorter than
             min_chars (120), so 414 sentences of 42 abstracts were never indexed; one is the
             gold abstract of test claim 1290 (every variant still has it in the top ten).
             With keep_tail: 21,133 passages instead of 21,015.
             nDCG@10, keep_tail vs reported (paired bootstrap):
               bm25 0.6562 -> 0.6562 (same on 300/300)   dense 0.6789 -> 0.6780 (p = 0.63)
               fusion 0.7097 -> 0.7097 (same on 300/300) fusion+rerank 0.7154 -> 0.7152 (p = 0.61)
               fine-tuned dense 0.7416 -> 0.7416         best system 0.7587 -> 0.7586 (p = 0.63)
Latency:     main ablation 6 min 36 s (dense index rebuilt: 141 s); best system 2 min 40 s

Observation: The bug is real but its effect is negligible: no comparison changes and no
             difference is significant. The reported numbers keep the original behaviour so
             they stay reproducible; keep_tail becomes the default after the assessment.
Next:        RESULTS.md "Problems found"; viva answer; issue #45 stays the record.

---

### 2026-10-13 — The improvement on a second dataset: NFCorpus (BEIR)
Commit:      84e0efd (main with finetune_dense.py --max-pairs-per-query, PR #59)
Config:      changed: dataset.hf_id=BeIR/nfcorpus, dataset.name=nfcorpus; otherwise unchanged
Command:     run_ablation.py (standard, results/nfcorpus); finetune_dense.py --name
             all-MiniLM-L6-v2-nfcorpus --epochs 12 --max-pairs-per-query 3; tune_fusion.py with
             the fine-tuned model; run_ablation.py --variants dense fusion with w = 0.2
             (results/nfcorpus_finetuned__fusion_weighted) and --variants fusion+rerank
             (..._rerank); the SciFact encoder with w = 0.3 (results/nfcorpus_scifact_encoder__
             fusion_weighted); scripts/significance_nfcorpus.py
Hardware:    NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)

Result:      Corpus 3,633 abstracts -> 16,574 passages; 323 test queries, 38.2 judged each
             (grades 1-2); training: 2,590 queries (2,470 train / 120 dev), 42.7 judged each,
             all grade 1, first 3 per query used -> 6,917 pairs.
             Fine-tuning: dev nDCG@10 0.3597 -> 0.3732 (best epoch 8 of 12).
             Weight on dev: 0 .3732, 0.1 .3832, 0.2 .3850 (chosen), 0.3 .3833, 0.5 .3757,
             0.7 .3648, 1.0 .3567, 1.5 .3373, 2.0 .3345.
             Test nDCG@10 (P@10, R@100, ms):
               BM25 0.312 (0.227, 0.201, 12)        dense 0.330 (0.251, 0.252, 13)
               fusion w=1 0.340 (0.253, 0.254, 31)  fusion+rerank 0.358 (0.260, 0.254, 168)
               fine-tuned dense 0.336 (0.258, 0.293, 13)
               improved, w=0.2 0.350 (0.267, 0.293, 29)   + rerank 0.358 (0.266, 0.293, 171)
               SciFact encoder, w=0.3 0.343 (0.253, 0.253, 29)
             Paired bootstrap: improved vs standard pipeline -0.008, p = 0.35; improved vs
             fine-tuned dense +0.014, p < 0.0001; fine-tuned vs pretrained dense +0.006,
             p = 0.49; improved + rerank vs improved +0.008, p = 0.23; SciFact encoder vs
             standard fusion +0.003, p = 0.56. Standard: rerank vs fusion +0.019, p = 0.0045.
Latency:     fine-tuning 18 min 23 s (1,102.7 s); weight search ~1 min; each test run ~1 min

Observation: The fusion half of the improvement transfers (weight again below the
             break-even 0.381; R@100 of the improved retriever equals the encoder's, as
             Proposition 1 predicts), and the reranker is again unnecessary. Fine-tuning helps
             little: lay queries and weak, same-grade labels. The improvement matches rather
             than beats the reranked pipeline on NFCorpus, at a sixth of the time.
Next:        full paper Section 5.4 and RQ4; RESULTS.md step 15.

---

<!-- add new entries above this line, newest last -->
