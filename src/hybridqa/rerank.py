"""Cross-encoder reranking.

A bi-encoder must compress a passage into one vector before it ever sees the query.
A cross-encoder concatenates them — [CLS] query [SEP] passage [SEP] — and runs full
self-attention across both, so every query token can attend to every passage token.
That is far more accurate and far more expensive: it is O(candidates) forward passes
per query, which is why it only ever runs over the fused top-N, never the corpus.

`candidates` in the config is the accuracy/latency dial the paper measures.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Run = List[Tuple[str, float]]


class CrossEncoderReranker:
    def __init__(self, cfg, device: str | None = None):
        self.model_name = cfg.get_path("rerank.model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.batch_size = cfg.get_path("rerank.batch_size", 64)
        self.max_length = cfg.get_path("rerank.max_length", 512)
        self.candidates = cfg.get_path("rerank.candidates", 50)
        self.top_k = cfg.get_path("rerank.top_k", 10)
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Run,
        chunk_texts: Dict[str, str],
        candidate_depth: int | None = None,
        top_k: int | None = None,
    ) -> Run:
        """Score the top-`candidate_depth` fused candidates and re-sort them.

        Candidates below the depth are appended after the reranked block with their
        original ordering, so recall at large k is never *reduced* by reranking.
        """
        depth = candidate_depth or self.candidates
        top_k = top_k or self.top_k
        head, tail = candidates[:depth], candidates[depth:]
        if not head:
            return []

        pairs = [(query, chunk_texts[cid]) for cid, _ in head]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked = sorted(
            ((cid, float(s)) for (cid, _), s in zip(head, scores)),
            key=lambda kv: -kv[1],
        )
        # keep the tail so downstream Recall@100 remains comparable across variants
        min_score = reranked[-1][1] if reranked else 0.0
        tail_scored = [(cid, min_score - 1e-3 * (i + 1)) for i, (cid, _) in enumerate(tail)]
        merged = reranked + tail_scored
        return merged[:top_k] if top_k else merged

    def rerank_batch(
        self,
        queries: Sequence[str],
        runs: Sequence[Run],
        chunk_texts: Dict[str, str],
        candidate_depth: int | None = None,
        top_k: int | None = None,
    ) -> List[Run]:
        return [
            self.rerank(q, run, chunk_texts, candidate_depth, top_k)
            for q, run in zip(queries, runs)
        ]
