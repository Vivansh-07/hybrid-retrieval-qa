# Viva preparation

Answer each of these out loud, without notes, before the viva. The answers below are
scaffolding — say them in your own words, and replace every bracketed placeholder with
your actual number. An examiner can tell the difference between a memorised answer and
an understood one, and the tell is usually that you cannot handle the follow-up.

---

## Tier 1 — You will almost certainly be asked these

**Why not just use BM25? / Why not just use the dense model?**
Each fails on a different query type. BM25 matches surface terms, so it is strong on
exact identifiers, gene names and numerals, and it fails when the claim is phrased
differently from the abstract. The dense bi-encoder matches meaning, so it survives
paraphrase, but it compresses a passage into one [384]-dimensional vector and rare
identifiers get averaged away — and on SciFact it is out of domain, which BEIR shows
costs dense models a lot. My ablation shows [X] queries where one succeeds and the
other fails; fusion recovers most of them.

**What is Reciprocal Rank Fusion and why not just average the scores?**
RRF scores a document as the sum over systems of 1/(k + rank). It uses ranks, not
scores. Averaging scores does not work because BM25 scores are unbounded and depend on
corpus statistics while cosine similarity lives in [-1, 1] — there is no fixed weighting
that transfers across queries. I did implement score fusion as a baseline; it scored
[Y] against RRF's [Z].

**What does k=60 do?**
It damps the top ranks. With k=60, rank 1 contributes 1/61 and rank 2 contributes 1/62 —
almost the same. A small k would make rank 1 hugely more valuable than rank 2, letting one
over-confident system dominate. 60 is Cormack et al.'s value; I swept k over [10, 30, 60,
100] and the result moved by less than [W], so the finding is not a k artefact.

**Bi-encoder vs cross-encoder — what is the actual difference?**
A bi-encoder encodes query and passage separately, so the only interaction is a dot
product between two fixed vectors. That lets you precompute the whole corpus offline.
A cross-encoder feeds `[CLS] query [SEP] passage [SEP]` through one Transformer, so every
query token attends to every passage token. Far more accurate, but it needs one forward
pass per pair — you cannot precompute anything, which is exactly why it runs only over
the fused top-[50] and never over the corpus.

**Why nDCG as the primary metric?**
It is the only one of my four that is both graded and rank-discounted. Recall@k ignores
ordering entirely — a reranker can improve the ranking a lot and not move Recall@10 at
all. Precision@k ignores grades. MRR only looks at the first hit. nDCG@10 divides DCG by
the ideal DCG, so it is comparable across queries with different numbers of gold docs.

**Walk me through what happens to one query, end to end.**
Practise this on a real query from your results with the actual retrieved passages open.
Tokenise → BM25 over the inverted index → top-100 chunks; embed the query, FAISS inner-
product search → top-100 chunks; RRF merges to a fused list; cross-encoder scores the top
50 pairs and re-sorts; the top 5 chunks go into the prompt numbered [1]–[5]; the reader
generates an answer with bracketed citations; citations map back to parent doc ids.

---

## Tier 2 — Likely, and where people get caught

**Where is the deep learning in this?** (The single most common trap in a DL course viva.)
Two learned Transformer components and one reader. The bi-encoder is a Siamese BERT
trained with a contrastive objective so semantically similar text lands nearby. The
cross-encoder is a BERT with a classification head fine-tuned on MS MARCO pairs. BM25 and
RRF are deliberately not learned — they are the classical baselines the learned components
have to beat, and that contrast is the point of the ablation.

**You did not train anything. Is this a deep learning project?**
Be straight about it: I used pretrained encoders and evaluated them under domain shift,
which is a legitimate zero-shot IR setting and is exactly what BEIR was built for. What I
contribute is the controlled comparison and the failure analysis. [If you did fine-tune,
say what, on what, for how long, and what it bought you.]

**Why chunk at all? SciFact abstracts are short.**
Honest answer: on SciFact the benefit is small — I measured [ΔX] nDCG@10 between chunked
and unchunked. Chunking matters because the pipeline is meant to generalise to long
documents where an embedding of 4,000 words is uselessly diffuse. I report both settings.

**Your chunks overlap. Does that not inflate your scores?**
No, because relevance is judged at document level, so I collapse chunk ranks to parent
doc ids before scoring and de-duplicate. Three chunks from the same document count once.
That is `chunk_to_doc` in `data.py`.

**How do you know the improvement is real and not noise?**
Paired bootstrap over per-query nDCG@10, 10,000 resamples, two-sided. Fusion+rerank vs
[baseline] gives p = [p]. [If any comparison is not significant, say so first — it reads
as rigour, not weakness.]

**What is the latency cost and would you deploy this?**
[A] ms/query for BM25, [B] for dense, [C] for fusion, [D] with reranking at depth 50.
Reranking is [D-C] ms of that. Whether it is worth it depends on the application: for
interactive search, reranking depth 25 is probably the right point on my curve; for
offline evidence gathering, take the full 100.

**What is FAISS actually doing? Is it approximate?**
I use `IndexFlatIP`, which is exact brute-force inner product — with [N] chunks × 384
dims that is a few milliseconds. FAISS's approximate indexes (IVF, HNSW) matter at
millions of vectors; at my scale approximation would trade accuracy for nothing.

**Why inner product rather than cosine?**
I L2-normalise the embeddings, after which inner product *is* cosine similarity. It is the
same thing with one fewer division per comparison.

---

## Tier 3 — Harder follow-ups

**When would fusion make things worse?**
When one retriever is much weaker than the other, RRF's uniform weights let the weak
system pull bad documents up. Weighted RRF helps, but the weights need a dev set. [Check
your own results — if you have queries where fusion underperforms both inputs, you have a
concrete example, and that is a strong answer.]

**Your reranker demoted a correct document. Why would that happen?**
Cross-encoders are trained on MS MARCO — web queries and passages. SciFact claims are
formal scientific statements, so there is a domain gap in the *reranker* too, not just the
retriever. `failure_analysis.py` counts these as `reranker_regression`; I found [N] of them.

**What would you do with more time / compute?**
In order of expected value: (1) fine-tune the bi-encoder on in-domain pairs, since the
domain gap is the dominant error source; (2) swap in a stronger reranker like monoT5-3B and
report the cost; (3) evaluate on a second BEIR dataset to test whether the fusion gain
generalises; (4) learned sparse retrieval (SPLADE) instead of BM25.

**How does this relate to RAG?**
This *is* the retrieval half of RAG, evaluated properly. Most RAG work reports only end
answer quality, which conflates retrieval failures with generation failures. Splitting the
evaluation is deliberate: if the gold passage was never retrieved, no reader can fix it.

**Is your citation metric measuring faithfulness?**
No, and I am careful about that in the paper. Citation precision measures whether the cited
documents are gold documents. It does not verify that the sentence is actually supported by
the cited passage — a model can cite the right document and still state something it does
not say. That is why I also annotated [50] answers manually for faithfulness.

**What is the biggest weakness of your work?**
Have a real answer ready; "nothing" is the worst possible response. Candidates: one dataset
so generality is unestablished; single annotator on the answer quality scores; document-level
judgements mean chunk-level precision is not directly measured; no in-domain fine-tuning.

---

## Code questions

Be able to open the file and explain, without preamble:

- `metrics.py` — the nDCG loop, and why IDCG is computed from sorted qrel values
- `fusion.py` — the RRF sum, and why the tie-break is on doc id (determinism)
- `rerank.py` — why the un-reranked tail is appended rather than discarded
- `data.py` — the sliding window indices, and `chunk_to_doc`
- `pipeline.py` — where the shared indexes are built and why they are built once

If you cannot explain a line, delete it or rewrite it in a way you can. Code you cannot
defend is worse than code you do not have.

---

## The day before

- Rerun the ablation from a clean clone. Numbers in your slides must match `results/`.
- Print the paper and one page with the ablation table on it. Bring both.
- Know your three headline numbers cold: nDCG@10 for the full system, the gain over the
  best single retriever, and the reranking latency.
