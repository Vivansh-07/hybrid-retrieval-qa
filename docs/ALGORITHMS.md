# The algorithms, and why each one

Your teacher asked that you know every algorithm in the project and why it was chosen.
This page covers them in pipeline order (Fig. 1 of the paper). For each: **what it does**,
**the idea in plain words**, **why this and not the alternative**, **the settings**, and
**where it is in the code**. The last line of each section is a one-sentence answer for
the viva.

Settings are in `configs/scifact.yaml`; the numbers are from `results/`.

---

## 0. Sentence-window chunking (corpus preparation)

**What.** Split every abstract into overlapping passages of 3 sentences, moving 2 sentences
at a time, with the paper's title in front. 5,183 abstracts → 21,015 passages.

**Idea.** Overlap means a fact that spans two sentences is still whole in some passage.
The title carries the topic, so every passage keeps it.

**Why not whole abstracts?** The dense encoder reads at most 256 tokens; 71.6% of the
abstracts are longer and would be cut off (only 1.2% of passages are). With whole
abstracts, dense retrieval drops 0.031 nDCG@10 (p = 0.009). Rankings are mapped back to
abstracts afterwards, because SciFact judges relevance per abstract.

**Settings.** `chunking.window_sentences: 3`, `stride_sentences: 2`, `min_chars: 120`.
**Code.** `src/hybridqa/data.py`: `chunk_document`, `build_chunks`, `chunk_to_doc`.

> Viva: "Short overlapping passages keep each one inside the encoder's 256-token limit,
> and I map them back to abstracts for scoring."

---

## 1. BM25 (sparse retrieval, Stage 1)

**What.** Classic keyword search over an inverted index.

**Formula in words.** For each word of the claim that also occurs in the passage, add:
*how rare the word is in the corpus* (IDF) × *how often it occurs in this passage*, where
the second factor saturates (the 10th repetition counts far less than the 1st, controlled by
`k1`) and is normalised for passage length (longer passages are penalised, controlled by
`b`). Words not in the claim contribute nothing.

**Why.** Exact matching is ideal for gene, drug and disease names, numbers and codes such
as "IL-6", and BM25 is the robust zero-shot baseline in BEIR. It needs no training.

**Weakness.** It cannot match different words with the same meaning: a claim about
"aspirin" does not find an abstract that only says "COX inhibitors".

**Why not SPLADE (learned sparse)?** It needs training and a larger index; BM25 keeps the
project simple and is the standard reference point.

**Settings.** `k1 = 0.9`, `b = 0.4` (BEIR's defaults), lower-casing, stop-word removal,
Porter stemming, a tokeniser that keeps "IL-6" whole; top 100 passages.
**Code.** `src/hybridqa/sparse.py`: `tokenize`, `BM25Retriever`.

> Viva: "BM25 scores word overlap, weighting rare words more and saturating repeated ones;
> it is precise for names and codes but blind to synonyms."

---

## 2. Dense bi-encoder with FAISS (dense retrieval, Stage 2)

**What.** A small Transformer (all-MiniLM-L6-v2, 6 layers, 22.7M parameters) turns the claim
and every passage into a 384-number vector; the passages whose vectors point in the most
similar direction (cosine similarity) are returned.

**Idea.** Texts with the same meaning get nearby vectors even when they share no words.

**Why a *bi*-encoder?** Claim and passage are encoded separately, so all 21,015 passage
vectors are computed once, offline, and a query costs one encoding plus a vector search
(about 20 ms). A cross-encoder would need 21,015 forward passes per claim.

**Why MiniLM?** Small and fast enough for a 4 GB laptop GPU, and a standard
sentence-transformers baseline. The larger BGE-small was tried too: 0.723 vs 0.679.

**Why FAISS flat (exact) search?** With 21,015 vectors, exact search is fast enough, so
there is no approximation error to explain.

**Weakness.** Trained on general web text, so specialised vocabulary is blurred; claim and
passage never "look at" each other.

**Settings.** `max_seq_length: 256`, normalised vectors, `index: flat_ip`, top 100.
**Code.** `src/hybridqa/dense.py`: `DenseRetriever` (`encode`, `index`, `search`).

> Viva: "Each text becomes a 384-dimensional vector; similar meaning means similar
> direction, so I can precompute the passages and search them in milliseconds."

---

## 3. Contrastive fine-tuning of the encoder (in-batch negatives)

**What.** Further training of MiniLM on SciFact's 689 training claims (120 more held out as
a development set), so it learns this domain's language.

**Loss in words.** In a batch of 32 (claim, correct passage) pairs, every claim is compared
with all 32 passages. The loss is a softmax cross-entropy that pushes each claim towards its
own passage and away from the other 31, which act as free "negative" examples. Similarities
are multiplied by a scale of 20 so the softmax is sharp enough to learn from (cosines lie
between -1 and 1).

**Why fine-tune?** The domain gap is the dense model's weakness, and SciFact provides
labelled training claims. Result: dense 0.679 → 0.742 (p < 0.0001), no extra cost per query.

**Why not an unsupervised method like GPL?** GPL is for domains without labelled queries;
we have 809 labelled claims, so supervised fine-tuning is the direct route.

**No leakage.** Test claims are never loaded; the number of epochs and the fusion weight
were chosen on the 120 development claims only. A fresh clone reproduces it exactly.

**Settings.** 12 epochs, batch 32, AdamW (learning rate 2e-5, weight decay 0.01), 10%
warm-up then linear decay, gradient clipping at 1; best development epoch kept (12).
**Code.** `scripts/finetune_dense.py`: `contrastive_loss` and the training loop.

> Viva: "Each claim must pick its own evidence passage out of the 32 in the batch; the
> other 31 are the negatives. That adapts the encoder to scientific wording."

---

## 4. Reciprocal Rank Fusion (Stage 3)

**What.** Merge the BM25 list and the dense list into one ranking.

**Formula.** score(d) = Σ over retrievers of w / (k + rank of d in that retriever's list),
with k = 60. A passage near the top of either list, or fairly high in both, ends up on top.

**Why ranks, not scores?** BM25 scores are unbounded (5, 20, 40…) and cosine scores lie in
[-1, 1]; adding them would let one dominate by scale. Ranks are comparable and need no
training or normalisation. (Min-max score fusion was tried: 0.717 vs 0.710, tied.)

**What k does.** It softens the gap between rank 1 and rank 2 so that one retriever's top
result does not decide everything; 10 to 100 all gave 0.709 to 0.714.

**The weight insight (our novelty).** With BM25 weight w = 0.3, a passage found *only* by
BM25 scores at most 0.3/61 = 0.0049, less than the dense list's 100th passage (1/160 =
0.0063). Fusion keeps 100 passages, so BM25-only passages never get in: BM25 can only
re-order the dense candidates. Break-even weight = 61/160 ≈ 0.38.

**Settings.** `k: 60`, `weights: [1, 1]` by default, `[0.3, 1]` in the best system (chosen
on development claims), keep top 100.
**Code.** `src/hybridqa/fusion.py`: `reciprocal_rank_fusion`, `normalized_score_fusion`, `fuse`.

> Viva: "RRF adds up 1/(60 + rank) from each list, so it combines rankings without
> comparing incompatible scores."

---

## 5. Cross-encoder reranking (Stage 4)

**What.** Rescore the 50 best fused passages with a model (ms-marco-MiniLM-L-6-v2) that reads
claim and passage *together* as one input: `[CLS] claim [SEP] passage [SEP]`.

**Idea.** Every claim word can attend to every passage word, so it judges relevance more
precisely than comparing two separate vectors.

**Why only 50?** It needs one forward pass per (claim, passage) pair; nothing can be
precomputed. More candidates cost more time but gave no more accuracy (N = 10 to 100:
0.716 to 0.722; latency 120 → 470 ms).

**What we found.** No significant gain after fusion in any configuration (+0.006, p = 0.65),
at almost 3× the latency, even with a 12× larger reranker. It was trained on web search
(MS MARCO), not science. That is why the best system skips it.

**Why not an LLM reranker (RankGPT)?** Far slower and costlier; recent work shows
cross-encoders remain competitive.

**Settings.** `candidates: 50`, inputs up to 512 tokens.
**Code.** `src/hybridqa/rerank.py`: `CrossEncoderReranker.rerank`.

> Viva: "A cross-encoder reads claim and passage jointly, which is accurate but too slow for
> the whole corpus, so it only rescores the top 50, and here it didn't pay off."

---

## 6. The reader: flan-t5 with post-hoc citation, and an extractive baseline (Stage 5)

**What.** flan-t5-base reads the top 5 passages and writes one sentence on whether they
support the claim; the sentence is then credited to the passage it shares the most words
with (post-hoc attribution). The extractive baseline (roberta-base-squad2) copies a span
from the top passage instead.

**Why "explain, then cite"?** Asked to answer and cite inline, flan-t5 replied with a bare
"[3]." for 49 of 50 claims (flan-t5-large: 47). The prompt was chosen on 20 training claims.

**Trade-off found.** Generative: more often correct (0.84 vs 0.48 on a 0–2 scale) but a
third of its answers say something the passage doesn't. Extractive: always faithful but
mostly fragments. Neither abstains when the evidence is missing.

**Settings.** `n_passages: 5`, greedy decoding, `max_new_tokens: 128`.
**Code.** `src/hybridqa/answer.py`: `AnswerGenerator`, `attribute`.

---

## 7. Evaluation: nDCG@10, Recall@k, MRR and the paired bootstrap

**nDCG@10.** DCG = Σ over the top 10 of (relevant ? 1 : 0) / log2(rank + 1); divide by the
best possible DCG for that claim. It rewards finding the evidence *and* ranking it high, which
is why it is the main metric (and BEIR's).

**Recall@k.** Share of a claim's relevant abstracts found in the top k. **MRR@10.** 1 / rank
of the first relevant abstract.

**Paired bootstrap (significance).** Resample the 300 claims with replacement 10,000 times,
recompute the mean difference between two systems each time, and ask how often a difference
at least as large would appear if there were no real difference. "Paired" because both
systems are scored on the same claims, so claim difficulty cancels out. p < 0.05 =
significant.

**Code.** `src/hybridqa/metrics.py`: `ndcg_at_k`, `recall_at_k`, `reciprocal_rank`,
`paired_bootstrap`; `src/hybridqa/significance.py`: `compare_runs`. Hand-checked in
`tests/test_metrics.py` (`test_ndcg_hand_computed`).

---

## 8. The abstention check (AUC and threshold)

**What.** Can a retrieval score tell when the evidence never reached the reader?

**How.** For each claim, take the top cross-encoder score and whether a gold abstract is in
the 5 passages. **AUC** = the probability that a claim whose evidence was retrieved has a
higher score than one whose evidence was not (0.5 = guessing, 1 = perfect): 0.81. A threshold
chosen on the development claims catches 68 of 74 misses but refuses 109 of 226 answerable
claims, so a score alone is not enough.

**Code.** `src/hybridqa/abstention.py`, `scripts/abstention.py`.
