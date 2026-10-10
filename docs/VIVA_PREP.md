# Viva preparation

Answer each of these out loud, without notes, before the viva. The answers below are
scaffolding — say them in your own words, and replace every bracketed placeholder with
your actual number. An examiner can tell the difference between a memorised answer and
an understood one, and the tell is usually that you cannot handle the follow-up.

For "explain this algorithm" questions, [ALGORITHMS.md](ALGORITHMS.md) has every
algorithm with the idea, why it was chosen and a one-sentence answer.

---

## Your teacher's checklist (7 Oct): novelty, survey, algorithms

**What is novel in your work?**
It is an empirical study, so the novelty is in what it shows, not a new model. Four things
the recent hybrid-retrieval papers (2022–2026) don't do:
(1) explain *per claim* which retriever wins: the word overlap between claim and evidence
decides it (dense wins on paraphrased claims, BM25 on claims worded like the evidence);
(2) show what a fusion weight does: below a BM25 weight of 61/160 ≈ 0.38, BM25 can only
re-order the dense candidates, never add new ones, which explains how down-weighting avoids
the "weakest link" effect that Wang et al. (PVLDB 2026) observed;
(3) a cheaper alternative to reranking: fine-tuning a 22M-parameter encoder on 689 claims
and fusing at a tuned weight gives 0.759, better than the reranked pipeline (0.715) at about
a third of the latency; and (4) a measured abstention baseline: the reranker's score has
AUC 0.81 but no cheap threshold. Say it as "what this paper adds", not "the first ever".

**What does the latest research say, and where does your work fit?**
(`docs/LITERATURE_SURVEY.md`, 40 references, 25 from 2022–2026.) Recent studies agree
hybrid retrieval helps, but the best design depends on the data: a 2026 PVLDB study across 11
datasets found no single best configuration and a "weakest link" effect; a 2026 benchmark on
financial documents found hybrid plus reranking best and BM25 beating dense there; DAT (2025)
sets the fusion weight per query with an LLM. In scientific QA, OpenScholar (*Nature* 2026)
answers from 45 million papers with expert-level citation accuracy. My work isolates the
retrieval stage on scientific claims and explains *why* fusion helps or hurts.

**How did you do the survey? Did you use the tools I suggested?**
Searched Semantic Scholar, arXiv, the ACL Anthology and Crossref for 2022–2026 work; read
each abstract; checked every reference against the publisher. Then say honestly which of
ResearchRabbit, Consensus, SciSpace and NotebookLM *you* used and what they added. Also:
three PhD theses from Waterloo's IR group, which produced BEIR (Thakur 2026, Lin 2024,
Pradeep 2025).

**Why Springer format?**
The course allows IEEE (6 pages) or Springer (12 pages). Springer's single column gives the
diagram and tables room; the body is 12 pages and the references follow.

---

## Tier 1 — You will almost certainly be asked these

**Why not just use BM25? / Why not just use the dense model?**
Each fails on a different query type. BM25 matches surface terms, so it is strong on
exact identifiers, gene names and numerals, and it fails when the claim is phrased
differently from the abstract. The dense bi-encoder matches meaning, so it survives
paraphrase, but it compresses a passage into one 384-dimensional vector and rare
identifiers get averaged away — and on SciFact it is out of domain. In my results the two
are statistically tied (nDCG@10 0.656 vs 0.679, p = 0.25), but they succeed on different
claims: for 42 of the 300 claims only one of them puts a gold paper in the top 10 (17 only
BM25, 25 only dense). Fusion keeps the gold paper in the top 10 for 27 of those 42, and is
significantly better than both (0.710; p < 0.0001 vs BM25, p = 0.019 vs dense).

**What is Reciprocal Rank Fusion and why not just average the scores?**
RRF scores a document as the sum over systems of 1/(k + rank). It uses ranks, not
scores. Adding raw scores does not work because BM25 scores are unbounded and depend on
corpus statistics while cosine similarity lives in [-1, 1]. The alternative is to min-max
normalise each list per query and then add them. I implemented that as a baseline: it
scored 0.717 against RRF's 0.710, and the difference is not significant (p = 0.27). So I
do not claim RRF is more accurate. I keep it because it needs no normalisation and no
tuning data, and because it is stable: the k sweep below barely moves the result.

**What does k=60 do?**
It damps the top ranks. With k=60, rank 1 contributes 1/61 and rank 2 contributes 1/62 —
almost the same. A small k would make rank 1 hugely more valuable than rank 2, letting one
over-confident system dominate. 60 is Cormack et al.'s value; I swept k over 10, 30, 60
and 100 and nDCG@10 stayed between 0.709 and 0.714 (a range of 0.005), so the finding is
not a k artefact.

**Bi-encoder vs cross-encoder — what is the actual difference?**
A bi-encoder encodes query and passage separately, so the only interaction is a dot
product between two fixed vectors. That lets you precompute the whole corpus offline.
A cross-encoder feeds `[CLS] query [SEP] passage [SEP]` through one Transformer, so every
query token attends to every passage token. Far more accurate, but it needs one forward
pass per pair — you cannot precompute anything, which is exactly why it runs only over
the fused top-50 and never over the corpus. (In general it is more accurate; on my data
it was not significantly better than fusion alone — see the Tier 2 questions.)

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
I did train one: I fine-tuned the dense bi-encoder (all-MiniLM-L6-v2, 6 layers, 22M
parameters) on SciFact's 809 training claims, which share no claim with the 300 test
claims. 689 claims (792 claim-passage pairs) for training, 120 held out as a dev set. The
loss is contrastive (multiple-negatives ranking): for a batch of 32 pairs, each claim must
score its own evidence passage above the 31 other passages in the batch, which is a
cross-entropy over the similarity matrix with the diagonal as the target. AdamW, lr 2e-5,
12 epochs, about 10 minutes on an RTX 3050. Dev nDCG@10 rose from 0.691 to 0.770; I kept
the best dev epoch and only then evaluated on test: dense 0.679 → 0.742 (p < 0.0001).
The rest of the project uses pretrained models zero-shot, which is the standard BEIR
setting, and the contribution there is the controlled comparison and the failure analysis.

**How do you know the fine-tuning did not overfit or leak test data?**
Three things. The test split is never loaded by the training script. The number of epochs
was chosen on the dev set before the test set was evaluated, once. And the gain on dev
(+0.079) carries over to unseen test claims (+0.063). The honest caveats: it is one
training run with one seed, and the positives are weak labels (the chunk of the evidence
paper that the base model found most similar), not human-marked sentences.

**Could someone else get your 0.759?**
Yes, exactly. On 3 Oct the repository was cloned fresh from GitHub and the whole chain was
rerun: fine-tuning (11 minutes), choosing the BM25 weight on the dev claims, then the
test evaluation. Every epoch's dev score was identical, the weight search picked 0.3
again, and all 21 test metrics matched to the last digit. Training is seeded and was
deterministic on this GPU. Caveat if pressed: a different GPU or library version could
change the last decimals, since GPU arithmetic is not guaranteed bit-identical across
hardware.

**Why chunk at all? SciFact abstracts are short.**
I measured it. For BM25 it makes no difference (+0.002 nDCG@10 unchunked, p = 0.82). For
the neural models it matters: on whole abstracts the dense encoder loses 0.031 (p = 0.009)
and the full system loses 0.029 (p = 0.007). The reason is the input limit: 71.6% of
whole abstracts are longer than the encoder's 256 tokens and get cut off, against 1.2% of
my 3-sentence windows. The cost of chunking is a 4x larger index (32 MB vs 8 MB). Fusion
still beats both single retrievers without chunking, so that finding does not depend on it.

**Your chunks overlap. Does that not inflate your scores?**
No, because relevance is judged at document level, so I collapse chunk ranks to parent
doc ids before scoring and de-duplicate. Three chunks from the same document count once.
That is `chunk_to_doc` in `data.py`.

**How do you know the improvement is real and not noise?**
Paired bootstrap over per-query nDCG@10, 10,000 resamples, two-sided, for every pair of
variants. Say the non-significant ones first — it reads as rigour, not weakness:
- dense vs BM25: +0.023, p = 0.25 — not significant, they are tied.
- reranker vs fusion: +0.006, p = 0.65 — not significant (helps 52 claims, hurts 46).
- fusion vs BM25: +0.053, p < 0.0001; fusion vs dense: +0.031, p = 0.019 — significant.
- full system vs BM25: +0.059, p = 0.0002; vs dense: +0.037, p = 0.025 — significant.

**What is the latency cost and would you deploy this?**
About 80 ms/query for BM25, 22 for dense, 119 for fusion, 329 with reranking at depth 50,
so reranking adds roughly 210 ms. (BM25 is the slow retriever because the library scores
every passage in pure Python; the dense search is one encoder pass plus a matrix product.) These are single-laptop numbers and varied up to about
2x between identical runs, so I quote them as approximate. Would I deploy the reranker
here? No: on this data it almost triples the latency for a gain that is not significant, and
accuracy is flat from depth 10 to 100 (0.716-0.722). I would deploy fusion, and only add a
reranker trained on scientific text.

**What is FAISS actually doing? Is it approximate?**
I use `IndexFlatIP`, which is exact brute-force inner product — with 21,015 chunks × 384
dims (32 MB) that is a few milliseconds. FAISS's approximate indexes (IVF, HNSW) matter at
millions of vectors; at my scale approximation would trade accuracy for nothing.

**Why inner product rather than cosine?**
I L2-normalise the embeddings, after which inner product *is* cosine similarity. It is the
same thing with one fewer division per comparison.

---

## Tier 3 — Harder follow-ups

**When would fusion make things worse?**
When one retriever is much weaker than the other, RRF's uniform weights let the weak
system pull bad documents up. Weighted RRF helps, but the weights need a dev set. In my
results this is rare: fusion scores below BOTH inputs on 1 of 300 claims and above both on
25. It does lose some of what a single retriever found: of the 42 claims where only one
retriever had a gold paper in the top 10, fusion keeps it there for 27, so 15 drop out.
The one claim is 1259 (tamoxifen metabolism and genetic make-up, gold paper 24341590):
BM25 and dense both rank the gold paper first, but through different chunks of it
(`24341590#1` and `24341590#4`). RRF fuses chunks, not documents, so those two votes do
not add up, and a chunk of another paper that both lists rank fairly high
(`13069283#0`) overtakes them; the gold paper drops to rank 2. This is a limitation of
chunk-level fusion. The reranker puts it back at rank 1. Fusing at document level is
future work.

**Your reranker demoted a correct document. Why would that happen?**
Cross-encoders are trained on MS MARCO — web queries and passages. SciFact claims are
formal scientific statements, so there is a domain gap in the *reranker* too, not just the
retriever. Across all 300 claims the reranker improved nDCG@10 on 52 and lowered it on 46,
which is why its average gain (+0.006) is not significant. `failure_analysis.py` counts the
worst cases as `reranker_regression`; I found [N] of them. [Fill in after running it.]

**Did you try better models? What happened?**
Yes, and I report them next to the baseline, not instead of it. Swapping the dense encoder
for BGE-small raised dense retrieval from 0.679 to 0.723 nDCG@10 (p = 0.002), and dense then
clearly beats BM25 (p = 0.0003). But fusion no longer helps: 0.721, the same as dense alone.
With MiniLM the two retrievers were tied and fusion beat both; with BGE one retriever
dominates and equal-weight RRF adds nothing. So the fusion gain depends on the retrievers
being of similar strength. I also tried a reranker about 8x heavier (bge-reranker-base): 0.714,
no better than the small one (p = 0.86) and no better than dense alone, at about 1.3 s per
query. Two rerankers giving the same answer makes that finding robust.

**So which retriever helps which kind of question?** (This is your research question.)
I split the claims by how many of their words also appear in the evidence abstract. On
paraphrased claims (low overlap) the dense retriever beats BM25 by 0.073 (p = 0.022); on
high-overlap claims BM25 is at least as good, and that is where fusion gains (+0.052 over
dense, p < 0.0001). Fine-tuning helps almost only the paraphrased claims (+0.106, against
+0.016 on the rest), because it teaches the encoder the paraphrases that word matching
misses. One expectation failed: BM25 is NOT better on claims with numbers or codes; dense
is (0.715 vs 0.655, p = 0.031). Caveat to say yourself: overlap uses the gold abstract, so
it describes the claims; a retriever cannot compute it.

**How did you choose the fusion weight without cheating?**
On the 120 development claims of the training split that fine-tuning had held out. I
tried BM25 weights from 0 to 2 (dense fixed at 1), picked the best by nDCG@10 (0.3 for
both encoders), and only then ran the test claims, once. Weight 0 reproduced the
dense-only dev score from fine-tuning exactly, which shows it was the same split. Why it
works: once dense is much stronger, an equal BM25 vote pulls good results down on
paraphrased claims; a smaller vote keeps BM25's help on exact-wording claims.

**So what does a BM25 weight of 0.3 actually do?** (A sharp examiner will ask this.)
It only re-orders the dense retriever's candidates; it never adds new ones. Work it out
from the formula: a passage that only BM25 found scores at most 0.3/(60+1) ≈ 0.0049, but
the dense retriever's 100th passage scores 1/(60+100) = 0.00625, and fusion keeps the
top 100. So every BM25-only passage falls off the list. I checked it on the test claims:
no fused document is outside the dense retriever's list. The break-even weight is
61/160 ≈ 0.38. It costs little here: only 2 test claims have the gold abstract in BM25's
top 10 but not in the dense list. One is "Ivermectin is used to treat onchocerciasis.",
which BM25 ranks 1st and the best system loses completely (you can show it in the demo).

**What does the failure analysis show?** (Labels were assigned by an AI assistant from fixed
definitions: say so, and know three examples.)
Of the 40 hardest claims, 17 (42%) are not retrieval errors: the gold abstract does not state
the claim. Three contradictory claim pairs share one abstract (e.g. "PPAR-RXRs are inhibited /
activated by PPAR ligands", and the abstract never mentions PPAR). 11 (28%) are terminology
mismatch: "aspirin" vs an abstract that says only "COX inhibitors"; claim 690 expands "SFM"
as a syndrome when its malaria abstract means severe falciparum malaria. 8 (20%) are reranker
errors: for "Ivermectin is used to treat onchocerciasis" BM25 ranks the gold 1st, fusion 3rd,
the reranker 17th.

**How did you choose those 40? Isn't that cherry-picking?**
By a fixed rule, before any labelling (`scripts/failure_analysis.py`): every test claim gets
a priority of 2 × (1 − Recall@10 of the full pipeline), plus 1.5 if the reranker lost an
abstract that fusion had found, plus the gap between BM25's and dense's Recall@10; the 40
highest are analysed. So it is deliberately the *hardest* claims, not a random sample, and
the 42% describes hard failures, not all errors. The full pipeline has no relevant abstract
in its top ten for 46 of the 300 claims; 39 of them are among the 40. The 40th, claim 70
("Activation of PPM1D suppresses p53 function"), has two relevant abstracts: the pipeline
ranks one 1st and the reranker pushed the other out. The 7 complete misses left out are
missed by every variant, tie at the cut-off, and were cut by claim order. Weaknesses to
admit: one (AI-assisted) annotator, and the rule leans towards reranker losses.

**How good are the generated answers?**
Scored on 0-2 with a rubric fixed before generating (by the AI assistant; not blind). The
generative reader is right more often (correctness 0.84 vs 0.48) but a third of its answers
contain something the cited passage does not say (it called albendazole a myeloma drug); the
extractive reader never invents anything but mostly returns fragments. Neither abstains when
the evidence is missing. And the first prompt made flan-t5 answer with a bare "[3]." 49 times
out of 50: the fix was the instruction (explain, then cite afterwards), not a bigger model.

**Why not just make the reader say "not enough evidence" when retrieval is unsure?**
I tested that (`scripts/abstention.py`). The reranker's top score does carry signal: it
separates claims whose gold abstract reached the reader from those where it did not with
an AUC of 0.81 (0.5 would be guessing). But a single cut-off is expensive: the threshold
chosen on the dev claims catches 68 of the 74 test misses only by also refusing 109 of the
226 claims it could have answered. So abstention needs more than a score threshold, e.g. a
model trained to check whether the passages support the claim.

**What would you do with more time / compute?**
In order of expected value: (1) fine-tune the reranker in-domain too, since fine-tuning the
bi-encoder gave the largest gain and two off-the-shelf rerankers did not help; (2) mine hard
negatives for the bi-encoder instead of only in-batch negatives, and repeat over several seeds; (3) evaluate on a second BEIR dataset to test whether the fusion gain
generalises; (4) learned sparse retrieval (SPLADE) instead of BM25.

**How does this relate to RAG?**
This *is* the retrieval half of RAG, evaluated properly. Most RAG work reports only end
answer quality, which conflates retrieval failures with generation failures. Splitting the
evaluation is deliberate: if the gold passage was never retrieved, no reader can fix it.

**Does your fusion result agree with the literature?** (Bruch et al., ACM TOIS 2024)
Partly. They compared RRF with a convex combination of normalised scores over several
datasets and found RRF sensitive to its parameters and a *tuned* combination better. On
SciFact I find RRF insensitive to k (0.709 to 0.714 for k from 10 to 100) and tied with
an *untuned* min-max score fusion (0.717 against 0.710, p = 0.27). Not a contradiction:
one dataset against several, and I did not tune the score combination. Where we agree:
one fusion parameter can be tuned from few labelled examples. They tune the score
weight; I tuned the BM25 weight inside RRF on 120 dev claims, and it gave the best result.

**Why fine-tune on labels instead of an unsupervised method such as GPL?**
GPL (Wang et al., NAACL 2022) is for a domain without labelled queries: it generates
synthetic queries for the target documents and labels them with a cross-encoder. SciFact
comes with 809 labelled training claims, so supervised fine-tuning is the direct route.
GPL is what I would use for a new corpus with no training claims.

**Is your abstention check new?**
No, and the paper does not claim it is. Selective question answering (Kamath et al., ACL
2020) trains a separate calibrator that decides when a QA model should abstain. My check
is a simpler baseline: can one retrieval score decide it? It cannot cheaply, which is
consistent with their reason for training a calibrator.

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

**How do you know your code computes what you say it does?**
Three layers. (1) 88 unit tests: the metrics are compared with values worked out by hand
from their definitions, RRF with a hand-computed fusion, the AUC with scikit-learn's.
(2) External agreement: my BM25 gives 0.656 on SciFact where the BEIR paper reports
0.665, so the data loading and evaluation behave as expected. (3) GitHub runs all the
tests on every pull request, and other tests check that the paper's references resolve
and that its headline numbers still match the files in `results/`. If asked, open
`tests/test_metrics.py` and walk through `test_ndcg_hand_computed`.

**Why pull requests for a one-person project?**
Every change goes through a branch and a pull request, so it is tested on GitHub before
it reaches `main`, and the paper is compiled there too (that build found a table that
overflowed its column, which I fixed). It also leaves a reviewable history: what changed,
why, and that the tests passed at the time.

---

## The day before

- Rerun the ablation from a clean clone (done once on 3 Oct for the best system: identical).
  Numbers in your slides must match `results/`.
- Print the paper and one page with the ablation table on it. Bring both.
- Know your headline numbers cold. Zero-shot pipeline: full system 0.715 nDCG@10; fusion
  gives +0.031 over dense (p = 0.019) and +0.053 over BM25 (p < 0.0001); the reranker adds
  about +210 ms/query for a gain over fusion that is not significant (p = 0.65).
  Fine-tuned: dense 0.679 → 0.742 (p < 0.0001), at no extra query cost.
  Best system: fine-tuned dense + BM25 at weight 0.3 (chosen on dev claims): 0.759 nDCG@10,
  Recall@10 0.905; +0.043 over the original pipeline (p = 0.003).
