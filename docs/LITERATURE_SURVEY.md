# Literature Survey: Hybrid Sparse–Dense Retrieval for Scientific Question Answering

Vivansh Garg · Department of Computer Science and Engineering (Data Science), Manipal University Jaipur

Deep Learning course project · first version 13 September 2026, updated 7 October 2026

> The September version reviewed 15 works up to 2023. This update adds 25 works from
> 2022–2026 (including three PhD theses) and revises the research gap in their light.
> `paper/Literature_Survey.docx` is generated from this file.

## 1. Introduction, scope and method

Question answering over a specialised document collection is limited less by the answering
model than by retrieval: if the passage containing the evidence is never retrieved, no reader
can recover it. This survey covers the threads that bear on that problem for scientific claims:
lexical and dense retrieval, the evidence that neither dominates under domain shift, hybrid
retrieval and rank fusion, neural reranking, retrieval-augmented generation (RAG) for the
scientific literature, and systems that decide when not to answer.

**How the literature was found.** Searches in Semantic Scholar, arXiv, the ACL Anthology and
Crossref for: hybrid retrieval, BM25 and dense fusion, reciprocal rank fusion, cross-encoder and
LLM reranking, zero-shot and domain-adapted dense retrieval, scientific claim verification,
scientific literature search, and abstention in RAG; with an emphasis on 2022–2026. Included
are peer-reviewed papers and widely used preprints that study one of these components directly;
for each, the abstract was read and the bibliographic details (authors, venue, year, pages) were
checked against the publisher's record (ACL Anthology, Crossref, Springer, Nature). Recent PhD
theses were taken from the University of Waterloo's information retrieval group, which produced
BEIR. *The list is to be cross-checked by the author with ResearchRabbit (citation map) and
Consensus; any addition goes into this file.*

## 2. Lexical (sparse) retrieval

BM25, formalised within the probabilistic relevance framework by Robertson and Zaragoza [1],
remains the reference point for first-stage retrieval. It scores a passage by the query terms it
contains, weighting rare terms more and normalising for length, over an inverted index. Learned
sparse retrieval keeps the inverted index while learning term weights: SPLADE [2] predicts sparse
weights with vocabulary expansion.

Recent benchmarks show that BM25's standing depends on the domain. On financial documents with
text and tables, BM25 outperforms state-of-the-art dense retrieval [26]; on literature-search
questions about recent ML and NLP papers, dense retrievers beat BM25 by 24.8 points of Recall@5
[30]. A single "better" retriever does not exist across domains.

## 3. Dense retrieval

Sentence-BERT [3] established the bi-encoder used throughout this project: query and passage are
encoded separately and compared by a dot product. DPR [4] trained such encoders for open-domain
QA; Contriever [5] showed that contrastive pre-training without labels already beats BM25 on many
BEIR datasets; ColBERT [6] keeps one vector per token for finer matching at a higher index cost.

Three recent directions matter here. **Multi-function embedders**: BGE-M3 [21] produces dense,
sparse and multi-vector representations from one model, so a single encoder can serve hybrid
retrieval. **Domain-specific training**: MedCPT [22] trains a retriever and reranker on 255
million PubMed click logs and sets the state of the art on six biomedical retrieval tasks; GPL
[16] adapts a dense retriever to a new domain without labels by generating queries for its
documents. **Robustness**: Lin's PhD thesis [37] is devoted to building robust retrieval systems
from dense models. This project takes the supervised route that SciFact allows: it fine-tunes a
small bi-encoder on the dataset's labelled training claims.

## 4. Domain shift and benchmarks

BEIR [7] is the central result motivating hybrid retrieval: across 18 datasets, BM25 is a robust
zero-shot baseline that many dense models fail to beat. SciFact [8], one of the BEIR datasets and
the one used in this project, pairs 5,183 scientific abstracts with expert-written claims.
SciFact-Open [31] scales the corpus to 500,000 abstracts and finds that claim-verification systems
built on the small corpus lose at least 15 F1 there. BRIGHT [34]
shows the limit of surface matching: the leading MTEB embedder, at 59.0 nDCG@10 on standard
benchmarks, drops to 18.3 on reasoning-intensive queries. Thakur's PhD thesis [36] (BEIR's lead
author) collects benchmarks and evaluation methods for robust retrieval and RAG across domains.

## 5. Hybrid retrieval and rank fusion

If sparse and dense retrieval fail on different queries, combining them should recover both.
Chen et al. [17] showed this for zero-shot retrieval: a deep retriever degrades sharply under
domain shift while BM25 is more robust, yet the two are complementary even when the deep model is
the weaker one; a hybrid gains on average 20.4% over the deep model and 9.5% over BM25 on three
out-of-domain datasets.

How to combine them is the harder question, because the scores are not comparable. Reciprocal
Rank Fusion [9] fuses ranks, summing 1/(k + rank) with k = 60. Bruch et al. [10] compared RRF with
a convex combination of normalised scores and found RRF sensitive to its parameters and a tuned
combination better, with the single weight tunable from few examples.

The most recent studies examine the full design space. Wang et al. [24] evaluate four retrieval
paradigms and their combinations on 11 datasets and report a **"weakest link" phenomenon**: one
weak retrieval path can substantially degrade the fused result, so paths should be assessed
before fusion, and no configuration is best everywhere. Akarsu et al. [26] benchmark ten
strategies on a financial QA collection with paired bootstrap tests; hybrid retrieval followed by
neural reranking is best there. Rao et al. [25] fuse dense, sparse and graph signals on SciFact,
FiQA and NFCorpus and find that a compact MiniLM encoder (the model family used in this
project) outperforms the larger BGE-Large once LLM reranking is added. DAT [23] makes the fusion weight
query-specific: an LLM judges the top result of each retriever and sets the weight per query,
outperforming fixed weights.

## 6. Neural reranking

Nogueira and Cho [11] introduced the cross-encoder reranker: query and passage are read together
with full attention, which is accurate but costs one forward pass per candidate. monoT5 [12] casts
relevance as sequence-to-sequence generation.

Large language models are the newest rerankers. RankGPT [27] showed that instructed LLMs such as
GPT-4 can rank passages as well as or better than supervised methods, and that the ranking can be
distilled into a 440M model. Two follow-ups temper the excitement on cost. Déjean et al. [28] find
traditional cross-encoders "very competitive" with GPT-4 listwise reranking, with effectiveness
out of domain depending on the model and the reranking depth. Rank-DistiLLM [29] trains
cross-encoders that match LLM rerankers while being up to 173 times faster and 24 times more
memory-efficient. Pradeep's PhD thesis [38] covers open rerankers and scalable RAG evaluation.

## 7. Retrieval-augmented generation for the scientific literature

RAG [13] and Fusion-in-Decoder [14] condition generation on retrieved passages; Gao et al. [20]
survey the field's progression from "naive" to "modular" RAG. Attribution is the sharper
evaluation target: ALCE [15] measures whether citations support the generated text.

Scientific literature is now a flagship application. OpenScholar [32] answers scientific
questions from 45 million open-access papers with citation-backed answers; its 8B model beats
GPT-4o and PaperQA2 in correctness on a new expert-written benchmark, and its citation accuracy
matches human experts, whereas GPT-4o hallucinates citations 78–90% of the time. PaperQA2 [33], a
language-model agent, matches or exceeds subject-matter experts on literature search,
summarisation and contradiction detection. LitSearch [30] provides 597 realistic literature-search
questions. For claims specifically, MultiVerS [35] improves scientific claim verification with
weak supervision and full-document context. All of these systems depend on a first-stage
retriever, and SciFact-Open [31] shows how much claim verification degrades as the corpus grows.

## 8. Knowing when not to answer

A reader given the wrong passages should say so rather than answer. Selective question answering
[18] trains a calibrator that decides when a QA model should abstain under domain shift.
Self-RAG [19] trains a language model to decide when to retrieve and to critique the retrieved
passages and its own output with special "reflection" tokens. CRAG [39] adds a lightweight
retrieval evaluator whose confidence triggers corrective actions, such as discarding the
retrieved documents and searching elsewhere. Wen et al. [40] survey abstention in LLMs from the
perspectives of the query, the model and human values.

## 9. Comparative summary

**Table I. Foundational work (from the September survey)**

| Ref. | Approach | Core idea | Principal limitation |
|---|---|---|---|
| [1] | BM25 (sparse) | Term frequency with IDF weighting and length normalisation over an inverted index | Purely lexical: fails when query and source use different words |
| [2] | SPLADE (learned sparse) | Transformer predicts sparse term weights with expansion | Needs supervised training; larger index |
| [3] | Sentence-BERT | Siamese BERT; similar sentences close in embedding space | One vector per passage loses term-level evidence |
| [4] | DPR (dense) | Dual encoder trained contrastively on QA pairs | Gains are in-domain; degrades under domain shift |
| [5] | Contriever | Unsupervised contrastive pre-training | Rare identifiers still poorly represented |
| [6] | ColBERT | Late interaction over per-token embeddings | One vector per token: much larger index |
| [7] | BEIR | Zero-shot benchmark, 18 datasets | A benchmark; does not say how to combine retrievers |
| [9] | RRF | Sums 1/(k + rank) over systems, k = 60 | Uniform weights let a weak system pull poor documents in |
| [10] | Fusion-function analysis | Convex combination vs RRF | Combination needs labelled in-domain data to tune |
| [11] | monoBERT | Cross-encoder with full cross-attention | One forward pass per candidate: shallow pools only |
| [12] | monoT5 | Relevance as sequence-to-sequence generation | Larger, slower |
| [13] | RAG | Retriever + generator trained end to end | Reports end answers only, conflating the failure sources |
| [15] | ALCE | Benchmark for citation quality | Evaluates attribution, not which stage failed |

**Table II. Recent work, 2022–2026**

| Ref. | Year | Work | Setting | Key finding | Relation to this project |
|---|---|---|---|---|---|
| [17] | 2022 | Zero-shot hybrid retrieval (ECIR) | 5 datasets, 3 out of domain | Lexical and deep retrieval complementary even when deep is weaker; +20.4% over deep | Same premise; we test it on scientific claims with significance tests |
| [24] | 2026 | Hybrid-search trade-offs (PVLDB) | 4 paradigms, 11 datasets | "Weakest link": a weak path degrades fusion; no universal best | We show the mechanism for weighted RRF and its cost on SciFact |
| [26] | 2026 | BM25 to Corrective RAG | 10 strategies, financial QA, paired bootstrap | Hybrid + reranking best; BM25 beats dense on finance | Same method (ablation + bootstrap), different domain; reranking does not help on SciFact |
| [25] | 2025 | Rethinking hybrid retrieval | SciFact, FiQA, NFCorpus | Compact MiniLM beats BGE-Large with LLM reranking | Same model family and dataset; we fine-tune instead of adding an LLM |
| [23] | 2025 | DAT, dynamic fusion weight | RAG, LLM-judged weight per query | Per-query weights beat fixed ones | Our overlap analysis gives a cheap, measurable per-claim signal |
| [21] | 2024 | BGE-M3 (Findings of ACL) | Multilingual, long documents | One model gives dense, sparse and multi-vector scores | Alternative to running two retrievers |
| [22] | 2023 | MedCPT (Bioinformatics) | 255M PubMed click logs | State of the art on 6 biomedical IR tasks | Domain training at scale; we fine-tune on 689 claims |
| [16] | 2022 | GPL (NAACL) | Unlabelled target domains | Domain adaptation from generated queries | SciFact has labels, so we fine-tune directly |
| [27] | 2023 | RankGPT (EMNLP) | TREC, BEIR | Instructed LLMs rank competitively; distillable | LLM reranking: costlier than our cross-encoder |
| [28] | 2024 | Cross-encoders vs LLM rerankers | TREC DL, BEIR, LoTTE | Cross-encoders remain very competitive with GPT-4 | Consistent with our choice of a cross-encoder |
| [29] | 2025 | Rank-DistiLLM (ECIR) | MS MARCO | Distilled cross-encoders match LLMs, up to 173x faster | Cost of reranking matters; we measure it |
| [31] | 2022 | SciFact-Open (Findings of EMNLP) | 500K abstracts | Systems lose at least 15 F1 in the open setting | Retrieval is the bottleneck for claims |
| [35] | 2022 | MultiVerS (Findings of NAACL) | SciFact and others | Claim verification with weak supervision, full documents | Downstream of our retrieval |
| [30] | 2024 | LitSearch (EMNLP) | 597 literature-search questions | Dense beats BM25 by 24.8 Recall@5; LLM reranking +4.4% | Scientific retrieval benchmark; dense favoured there |
| [32] | 2026 | OpenScholar (Nature) | 45M papers, ScholarQABench | Citation accuracy at expert level; GPT-4o hallucinates citations 78–90% | Full scientific RAG; we isolate its retrieval stage |
| [33] | 2024 | PaperQA2 | Literature tasks, LitQA2 | Agent matches or exceeds experts | Same goal at agent scale |
| [34] | 2025 | BRIGHT (ICLR) | 1,384 reasoning queries | Best embedder falls from 59.0 to 18.3 nDCG@10 | Limits of surface and semantic matching |
| [20] | 2023 | RAG survey | — | Naive, advanced and modular RAG | Context |
| [19] | 2024 | Self-RAG (ICLR) | QA, fact verification | LM learns to retrieve on demand and critique | Trained abstention signal |
| [39] | 2024 | CRAG | 4 datasets | Retrieval evaluator triggers corrective actions | We test whether the raw retrieval score suffices |
| [18] | 2020 | Selective QA (ACL) | QA under domain shift | Trained calibrator decides when to abstain | Same question, simpler baseline in our study |
| [40] | 2025 | Abstention survey (TACL) | — | Framework: query, model, human values | Context |
| [36]–[38] | 2024–26 | PhD theses (Waterloo) | Robust retrieval, rerankers, RAG evaluation | — | Background reading |

## 10. Research gap and how this project addresses it

The recent literature confirms the premise of hybrid retrieval and shows how much the best
design depends on the data [17], [24], [26]. Four gaps remain open for scientific claims.

1. **Which retriever helps which claim is measured, if at all, with an LLM.** Recent studies
   report aggregate scores per configuration [24], [26]; adaptive methods such as DAT [23] set
   the weight per query with an LLM. A measurable property of the claim that predicts which
   retriever wins would give a cheap signal. *This project* splits the SciFact test claims by
   their word overlap with the evidence: dense retrieval wins on paraphrased claims, BM25 holds
   on claims that share the evidence's wording.

2. **The "weakest link" is observed but its mechanism in weighted fusion is not explained.**
   Wang et al. [24] show that a weak path degrades fusion, and fusion weights are usually tuned
   as a black box [10], [23]. *This project* shows from the RRF formula that below a BM25 weight
   of about 0.38 a document found only by BM25 can never enter the fused top 100, so fusion
   becomes a re-ordering of the dense candidates; on SciFact this costs 2 of 300 claims.

3. **Reranking is assumed to help, and its cost is rarely weighed against cheaper options.**
   Hybrid retrieval plus reranking is the strongest configuration on financial data [26], and
   LLM rerankers are increasingly used despite their cost [27]–[29]. *This project* finds that on
   scientific claims a cross-encoder adds no significant accuracy after fusion in any
   configuration, while fine-tuning the 22M-parameter bi-encoder on 689 training claims and
   fusing it with BM25 at a tuned weight gives the best result (nDCG@10 0.759) at about a third
   of the original pipeline's latency.

4. **Abstention is learned, and the simplest baseline is rarely quantified.** Selective QA,
   Self-RAG and CRAG train a calibrator, the LM or an evaluator [18], [19], [39]. *This project*
   measures how far the raw retrieval score goes: the top cross-encoder score predicts whether
   the evidence reached the reader with an AUC of 0.81, yet a threshold chosen on held-out
   claims refuses 109 of 226 answerable claims to catch 68 of 74 misses — evidence for trained
   evaluators.

The September survey's earlier gaps still hold and are addressed as planned: a controlled
four-way ablation (BM25, dense, fusion, fusion + reranking) over one corpus and index, accuracy
reported with paired significance tests and latency, and a categorisation of the remaining
failures.

## References

[1] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.
[2] T. Formal, B. Piwowarski, and S. Clinchant, "SPLADE: Sparse lexical and expansion model for first stage ranking," in *Proc. ACM SIGIR*, 2021, pp. 2288–2292.
[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992.
[4] V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, "Dense passage retrieval for open-domain question answering," in *Proc. EMNLP*, 2020, pp. 6769–6781.
[5] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave, "Unsupervised dense information retrieval with contrastive learning," *Transactions on Machine Learning Research*, 2022.
[6] O. Khattab and M. Zaharia, "ColBERT: Efficient and effective passage search via contextualized late interaction over BERT," in *Proc. ACM SIGIR*, 2020, pp. 39–48.
[7] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models," in *Proc. NeurIPS Datasets and Benchmarks Track*, 2021.
[8] D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van Zuylen, A. Cohan, and H. Hajishirzi, "Fact or fiction: Verifying scientific claims," in *Proc. EMNLP*, 2020, pp. 7534–7550.
[9] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in *Proc. ACM SIGIR*, 2009, pp. 758–759.
[10] S. Bruch, S. Gai, and A. Ingber, "An analysis of fusion functions for hybrid retrieval," *ACM Transactions on Information Systems*, vol. 42, no. 1, pp. 1–35, 2024.
[11] R. Nogueira and K. Cho, "Passage re-ranking with BERT," arXiv:1901.04085, 2019.
[12] R. Nogueira, Z. Jiang, R. Pradeep, and J. Lin, "Document ranking with a pretrained sequence-to-sequence model," in *Findings of EMNLP*, 2020, pp. 708–718.
[13] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, 2020.
[14] G. Izacard and E. Grave, "Leveraging passage retrieval with generative models for open domain question answering," in *Proc. EACL*, 2021, pp. 874–880.
[15] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling large language models to generate text with citations," in *Proc. EMNLP*, 2023, pp. 6465–6488.
[16] K. Wang, N. Thakur, N. Reimers, and I. Gurevych, "GPL: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval," in *Proc. NAACL-HLT*, 2022, pp. 2345–2360.
[17] T. Chen, M. Zhang, J. Lu, M. Bendersky, and M. Najork, "Out-of-domain semantics to the rescue! Zero-shot hybrid retrieval models," in *Advances in Information Retrieval (ECIR 2022)*, LNCS, Springer, 2022, pp. 95–110.
[18] A. Kamath, R. Jia, and P. Liang, "Selective question answering under domain shift," in *Proc. ACL*, 2020, pp. 5684–5696.
[19] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in *Proc. ICLR*, 2024.
[20] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, Q. Guo, M. Wang, and H. Wang, "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023.
[21] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, "M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation," in *Findings of ACL*, 2024, pp. 2318–2335.
[22] Q. Jin, W. Kim, Q. Chen, D. C. Comeau, L. Yeganova, W. J. Wilbur, and Z. Lu, "MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval," *Bioinformatics*, vol. 39, no. 11, btad651, 2023.
[23] H.-L. Hsu and J. Tzeng, "DAT: Dynamic alpha tuning for hybrid retrieval in retrieval-augmented generation," arXiv:2503.23013, 2025.
[24] M. Wang, B. Tan, Y. Gao, H. Jin, et al., "Balancing the blend: An experimental analysis of trade-offs in hybrid search," *Proceedings of the VLDB Endowment*, vol. 19, no. 8, pp. 1715–1730, 2026.
[25] A. Rao, H. Alipour, and N. Pendar, "Rethinking hybrid retrieval: When small embeddings and LLM re-ranking beat bigger models," arXiv:2506.00049, 2025.
[26] M. Akarsu, R. Karaman, and C. Mierbach, "From BM25 to corrective RAG: Benchmarking retrieval strategies for text-and-table documents," arXiv:2604.01733, 2026.
[27] W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren, "Is ChatGPT good at search? Investigating large language models as re-ranking agents," in *Proc. EMNLP*, 2023, pp. 14918–14937.
[28] H. Déjean, S. Clinchant, and T. Formal, "A thorough comparison of cross-encoders and LLMs for reranking SPLADE," arXiv:2403.10407, 2024.
[29] F. Schlatt, M. Fröbe, H. Scells, S. Zhuang, et al., "Rank-DistiLLM: Closing the effectiveness gap between cross-encoders and LLMs for passage re-ranking," in *Advances in Information Retrieval (ECIR 2025)*, LNCS, Springer, 2025, pp. 323–334.
[30] A. Ajith, M. Xia, A. Chevalier, T. Goyal, D. Chen, and T. Gao, "LitSearch: A retrieval benchmark for scientific literature search," in *Proc. EMNLP*, 2024, pp. 15068–15083.
[31] D. Wadden, K. Lo, B. Kuehl, A. Cohan, I. Beltagy, L. L. Wang, and H. Hajishirzi, "SciFact-Open: Towards open-domain scientific claim verification," in *Findings of EMNLP*, 2022, pp. 4719–4734.
[32] A. Asai, J. He, R. Shao, et al., "Synthesizing scientific literature with retrieval-augmented language models," *Nature*, vol. 650, pp. 857–863, 2026 (arXiv:2411.14199, "OpenScholar").
[33] M. Skarlinski, S. Cox, J. M. Laurent, J. D. Braza, et al., "Language agents achieve superhuman synthesis of scientific knowledge," arXiv:2409.13740, 2024.
[34] H. Su, H. Yen, M. Xia, W. Shi, et al., "BRIGHT: A realistic and challenging benchmark for reasoning-intensive retrieval," in *Proc. ICLR*, 2025.
[35] D. Wadden, K. Lo, L. L. Wang, A. Cohan, I. Beltagy, and H. Hajishirzi, "MultiVerS: Improving scientific claim verification with weak supervision and full-document context," in *Findings of NAACL*, 2022, pp. 61–76.
[36] N. Thakur, "Benchmarks, data, and evaluation for robust retrieval and retrieval-augmented generation on heterogeneous domains and languages," Ph.D. thesis, University of Waterloo, 2026.
[37] S.-C. Lin, "Building a robust retrieval system with dense retrieval models," Ph.D. thesis, University of Waterloo, 2024.
[38] R. Pradeep, "Democratizing and modernizing information access: From open rerankers to scalable RAG evaluation," Ph.D. thesis, University of Waterloo, 2025.
[39] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, "Corrective retrieval augmented generation," arXiv:2401.15884, 2024.
[40] B. Wen, J. Yao, S. Feng, C. Xu, et al., "Know your limits: A survey of abstention in large language models," *Transactions of the Association for Computational Linguistics*, vol. 13, pp. 529–556, 2025.
