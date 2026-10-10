"""Rank fusion.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):

    RRF(d) = sum over systems s of  w_s / (k + rank_s(d))

It combines RANKS, not scores, which is the point: BM25 scores are unbounded and
corpus-dependent while cosine similarities live in [-1, 1], so no fixed weighting
of the raw numbers transfers across queries. k (default 60) damps the influence of
the very top ranks so one over-confident system cannot dominate the fused list.

`normalized_score_fusion` is included as the baseline RRF is compared against in
the paper — min-max normalise each list, then take a weighted sum.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Run = List[Tuple[str, float]]  # ranked [(doc_id, score)], best first


def reciprocal_rank_fusion(
    runs: Sequence[Run],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> Run:
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError(f"got {len(runs)} runs but {len(weights)} weights")

    fused: Dict[str, float] = {}
    for run, w in zip(runs, weights):
        for rank, (doc_id, _score) in enumerate(run, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if top_k is None else ordered[:top_k]


def min_max_normalize(run: Run) -> Run:
    if not run:
        return []
    scores = [s for _, s in run]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [(d, 1.0) for d, _ in run]
    return [(d, (s - lo) / (hi - lo)) for d, s in run]


def normalized_score_fusion(
    runs: Sequence[Run],
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> Run:
    """Weighted sum of min-max normalised scores (the 'convex combination' baseline)."""
    if weights is None:
        weights = [1.0] * len(runs)
    fused: Dict[str, float] = {}
    for run, w in zip(runs, weights):
        for doc_id, score in min_max_normalize(run):
            fused[doc_id] = fused.get(doc_id, 0.0) + w * score
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if top_k is None else ordered[:top_k]


def fuse(runs: Sequence[Run], cfg) -> Run:
    """Config-driven dispatch."""
    if cfg.get_path("fusion.level", "passage") == "document":
        return fuse_documents(runs, cfg)
    method = cfg.get_path("fusion.method", "rrf")
    weights = cfg.get_path("fusion.weights")
    top_k = cfg.get_path("fusion.top_k", 100)
    if method == "rrf":
        return reciprocal_rank_fusion(runs, cfg.get_path("fusion.k", 60), weights, top_k)
    if method in {"nsf", "normalized", "convex"}:
        return normalized_score_fusion(runs, weights, top_k)
    raise ValueError(f"unknown fusion method: {method}")


def fuse_documents(runs: Sequence[Run], cfg) -> Run:
    """Fuse documents instead of passages.

    At passage level, when BM25 ranks passage #1 of a paper first and the dense
    retriever ranks passage #4 of the same paper first, RRF sees two different items
    and their votes do not add up. Here each list keeps only the best-ranked passage of
    every document, the lists are fused by DOCUMENT id, and each fused document is then
    represented by its best-ranked passage in either list, so the reranker and the reader
    still receive a passage.
    """
    doc_runs: List[Run] = []
    best: Dict[str, Tuple[int, str]] = {}  # doc_id -> (best rank in any list, passage id)
    for run in runs:
        seen, doc_run = set(), []
        for rank, (chunk_id, score) in enumerate(run):
            doc_id = chunk_id.split("#", 1)[0]
            if doc_id in seen:
                continue  # a lower-ranked passage of a document already in this list
            seen.add(doc_id)
            doc_run.append((doc_id, score))
            if doc_id not in best or rank < best[doc_id][0]:
                best[doc_id] = (rank, chunk_id)
        doc_runs.append(doc_run)

    passage_cfg = cfg.override("fusion.level", "passage")  # same method, k and weights, on documents
    return [(best[doc_id][1], score) for doc_id, score in fuse(doc_runs, passage_cfg)]
