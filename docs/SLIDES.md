# Slides and speaker notes

The deck (12 slides) is at <https://claude.ai/artifact/GEbH9cU6oLxKbpxAtDsCMC>. It is private: open it while logged in to
claude.ai. Its menu has **Present** and **Download** (PowerPoint or PDF); download a
copy before the presentation in case the classroom has no internet.

Below: each slide's title, what to say (the same text as the deck's speaker notes) and
where its numbers come from. Every number on the slides is taken from `results/`.
Say it in your own words; don't read it out.

Timing: about 45 seconds per slide, so 9 minutes for the slides, then 3 minutes of demo.

## 1. Hybrid Sparse–Dense Retrieval for Scientific Question Answering

Introduce yourself and the one question of the project: when a system answers a scientific question from the literature, which part of the retrieval actually helps, for which kind of question, and what does it cost in time? Everything you will show was measured on SciFact, a public benchmark of 300 scientific claims. The demo comes at the end.

## 2. A reader can only cite what retrieval finds

Systems that answer questions from documents first retrieve a few relevant passages, then write an answer from them with citations; the answer can only be as good as the retrieved evidence. There are two families of search. BM25 is keyword search: precise for names and numbers, blind to synonyms. Dense retrieval turns text into vectors so it can match meaning, but it was trained on general text. On SciFact the two are statistically tied overall but succeed on different claims, 73 against 60, which is why combining them should help. The example on the right is a real claim from my failure analysis: the claim says aspirin, its paper says COX inhibitors, and every method misses it.

*Numbers:* `results/failure_analysis.csv` (the aspirin claim); `results/significance.json` (dense vs BM25: +0.023, p = 0.25, better on 73 claims, worse on 60)

## 3. What recent work found (2022–2026)

I updated the literature survey to 2022 to 2026; it has 40 references, all checked against the publisher. Four findings matter for this project. First, hybrid retrieval helps because the two retrievers are complementary, but a 2026 PVLDB study across 11 datasets found a weakest-link effect: one weak retriever can drag fusion down. Second, rerankers keep getting more accurate but also more expensive; distilled cross-encoders match LLM rerankers while being up to 173 times faster. Third, DAT in 2025 sets the fusion weight per query with an LLM. Fourth, OpenScholar, published in Nature this year, answers scientific questions from 45 million papers with expert-level citations. The gap I address: nobody explains, claim by claim, why one retriever wins, or what a fusion weight actually does to the set of candidates.

*Numbers:* `docs/LITERATURE_SURVEY.md` (refs [17], [24], [27], [29], [23], [32])

## 4. SciFact: checking claims against 5,183 abstracts

SciFact is a public benchmark from the BEIR collection: 5,183 abstracts of biomedical papers and 300 test claims, each with the abstract or abstracts that support or refute it. I split every abstract into overlapping passages of three sentences with the title in front, which gives about 21,000 passages; results are mapped back to abstracts for scoring. The 809 training claims are never used for testing: I used them only to fine-tune the encoder, with 120 held out for early stopping. The main metric is nDCG at 10: it rewards putting the relevant abstract near the top of the first ten results. For every comparison I use a paired bootstrap test with 10,000 resamples, so I can say whether a difference is real or noise.

*Numbers:* BEIR SciFact; passage count from `results/index_build_timing/`; train/dev split from `results/finetune_all-MiniLM-L6-v2-scifact/history.json`

## 5. Five stages from claim to cited answer

Here is the whole system. A claim goes to both retrievers at once: BM25 and the dense retriever each return their top 100 passages. Reciprocal Rank Fusion merges the two lists using only the ranks, with the formula shown: a passage ranked high by either list, or fairly high by both, ends up on top. Ranks are used because BM25 scores and cosine similarities are on different scales. Then a cross-encoder, which reads the claim and each passage together, re-scores the top 50. Finally a small language model, flan-t5-base, writes an answer from the top five passages and cites them. Every stage can be switched off from one configuration file, which is what makes the ablation on the next slide possible.

*Numbers:* `configs/scifact.yaml`; `src/hybridqa/fusion.py`

## 6. Fusion helps; the reranker adds only latency

This is the controlled ablation: same corpus, same passages, same claims, only the retrieval strategy changes. BM25 alone scores 0.656, dense alone 0.679, and the two are statistically tied. Fusing them gives 0.710, significantly better than either one. Adding the cross-encoder reranker gives 0.715, but that extra 0.006 is not significant (p = 0.65) and it nearly triples the time per claim, from 119 to 329 milliseconds. So in this setting fusion is the stage that earns its cost and the reranker is not. Latency varied up to about two times between identical runs, so treat the milliseconds as approximate.

*Numbers:* `results/ablation.csv`, `results/significance.json`

## 7. Word overlap decides which retriever wins

To find out which retriever helps which claims, I split the 300 test claims by how many of their words appear in the evidence abstract, at the median. On the left, paraphrased claims: dense retrieval beats BM25 clearly, because it matches meaning. On the right, claims worded like their evidence: BM25 is at least as good as dense, and fusion gains most over dense. Fusion is never worse than the better of the two, which is the practical argument for running both. Note also how much harder paraphrased claims are: about 0.5 against about 0.9.

*Numbers:* `results/query_subsets.json` (split `overlap`)

## 8. Fine-tuning the encoder gave the largest gain

The biggest improvement did not come from adding stages but from adapting the dense encoder to the domain. I fine-tuned MiniLM on SciFact's 689 training claims with a contrastive loss, using 120 other training claims to decide when to stop; it took about ten minutes on my laptop GPU. Dense retrieval alone went from 0.679 to 0.742, better than a larger off-the-shelf encoder. Then I fused it with BM25, giving BM25 a weight of 0.3 that I chose on the held-out claims, never on the test set. The result is 0.759, significantly better than the original pipeline (p = 0.003), and since it needs no reranker it is also about three times faster. If asked what the weight does: at 0.3, BM25 can only re-order the dense retriever's candidates, never add new ones, because a passage only BM25 found scores less than the dense retriever's 100th passage.

*Numbers:* `results/dense_minilm_finetuned/comparison.json`, `results/dense_minilm_finetuned__fusion_weighted/`, `results/dense_bge_small/`, `results/fusion_weights_all-MiniLM-L6-v2-scifact.json`

## 9. 42% of the hardest misses are not retrieval errors

To see what still goes wrong, I examined the 40 test claims where the full pipeline fails most clearly and put each into one of six categories fixed in advance. The largest group, 17 claims, is not a retrieval error at all: the abstract marked as the answer does not actually state the claim. For example, two opposite claims about PPAR share one abstract that never mentions PPAR. Eleven are vocabulary mismatches, like a claim where the abbreviation SFM was expanded wrongly. Eight are cases where fusion had found the right abstract and the reranker pushed it out, which matches the reranker's lack of overall gain. Be open that the labels were assigned with AI assistance following fixed definitions. Before presenting, re-read these three claims in results/failure_analysis.csv so you can explain them in your own words.

*Numbers:* `results/failure_summary.json`, `results/failure_analysis.csv`

## 10. The generative reader is more useful but less faithful

Retrieval is only half of question answering, so I also scored the answers: 50 test claims, two readers, four criteria on a 0 to 2 scale. The generative reader writes real sentences and is right more often, but a third of its answers contain something the cited passage does not say. The extractive reader only copies text, so it is always faithful, but its answers are often fragments. Two practical lessons: the prompt wording mattered more than model size, since the first prompt produced a bare citation 49 times out of 50, and neither reader says "not enough evidence" when the right abstract was not retrieved. The scoring was done with AI assistance using a rubric written before scoring; mention that.

*Numbers:* `results/answer_quality.json`, `docs/ANSWER_RUBRIC.md`

## 11. What this project adds

This is what the project adds compared with the recent work on slide 3. It is an empirical study, so the novelty is in the findings. One: which retriever wins depends on how many words the claim shares with its evidence; dense wins on paraphrased claims. Two: I show from the fusion formula that below a BM25 weight of about 0.38, BM25 can only re-order what the dense model found, never add anything new; that explains how lowering the weight avoids the weakest-link effect. Three: fine-tuning a small encoder and fusing it at a tuned weight beats the reranked pipeline at a third of the cost. Four: a measured baseline for abstention, showing a retrieval score alone is not enough.

*Numbers:* `results/query_subsets.json`, paper Section V (weighted fusion), `results/dense_minilm_finetuned__fusion_weighted/`, `results/abstention.json`

## 12. Takeaways and limits

To sum up: combining keyword and meaning-based search helps when the two are of similar strength, because they fail on different claims. The cross-encoder reranker, the most expensive stage, gave no significant gain in any configuration. The largest gain came from fine-tuning the dense encoder on in-domain claims: 0.759 at about a third of the original pipeline's cost. The limits: one dataset; relevance judged per abstract; the failure labels and answer scores come from a single AI-assisted annotator; and the reranker was trained on web search. Next I would choose fusion weights per claim, try a reranker trained on scientific text, and make the reader abstain when evidence is missing. Now switch to the demo, already started with python scripts/demo.py --best, and use the four claim buttons.

*Numbers:* paper, Conclusion and Limitations
