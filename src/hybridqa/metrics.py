"""Retrieval metrics, implemented from the definitions so every number is defensible.

All metrics take a ranked list of DOCUMENT ids (already de-duplicated from chunk ids)
and the qrels for that query: {doc_id: graded_relevance}.

  Recall@k     |relevant retrieved in top-k| / |relevant total|   -- did we find the evidence?
  Precision@k  |relevant retrieved in top-k| / k                  -- how much noise came with it
  MRR          1 / rank of the first relevant document            -- how fast the user gets there
  nDCG@k       DCG@k / IDCG@k with DCG = sum rel_i / log2(i+1)    -- graded, rank-discounted

nDCG is the headline metric because it is the only one of the four that is both
graded and rank-sensitive, which is what a reranker actually changes.
`validate_against_pytrec_eval` cross-checks these against the standard IR toolkit.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

Qrels = Dict[str, Dict[str, int]]


def recall_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], relevant: Dict[str, int], k: int | None = None) -> float:
    cutoff = ranked[:k] if k else ranked
    for i, d in enumerate(cutoff, start=1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    gains = [float(relevant.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(v) for v in relevant.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Qrels,
    k_values: Sequence[int] = (1, 3, 5, 10, 20, 100),
) -> Dict[str, float]:
    """Macro-average every metric over the queries that have judgments.

    `run` is {query_id: [doc_id, ...]} ordered best-first.
    """
    qids = [q for q in run if q in qrels and qrels[q]]
    if not qids:
        raise ValueError("no evaluated queries: run and qrels share no query ids")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"Recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in qids) / len(qids)
        out[f"Precision@{k}"] = sum(precision_at_k(run[q], qrels[q], k) for q in qids) / len(qids)
        out[f"nDCG@{k}"] = sum(ndcg_at_k(run[q], qrels[q], k) for q in qids) / len(qids)
    out["MRR"] = sum(reciprocal_rank(run[q], qrels[q]) for q in qids) / len(qids)
    out["MRR@10"] = sum(reciprocal_rank(run[q], qrels[q], 10) for q in qids) / len(qids)
    out["num_queries"] = float(len(qids))
    return out


def per_query_metrics(
    run: Dict[str, List[str]],
    qrels: Qrels,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Per-query scores — the input to the failure analysis and to significance testing."""
    return {
        q: {
            f"Recall@{k}": recall_at_k(run[q], qrels[q], k),
            f"nDCG@{k}": ndcg_at_k(run[q], qrels[q], k),
            "RR": reciprocal_rank(run[q], qrels[q]),
        }
        for q in run
        if q in qrels and qrels[q]
    }


def paired_bootstrap(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    n_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap p-value for 'B beats A'.

    Report this next to any headline improvement: with 300 SciFact queries a
    +1 nDCG point difference is often inside the noise, and a viva examiner is
    entitled to ask whether the gain is real.
    """
    import random

    qids = sorted(set(scores_a) & set(scores_b))
    if not qids:
        return 1.0
    diffs = [scores_b[q] - scores_a[q] for q in qids]
    observed = sum(diffs) / len(diffs)
    centered = [d - observed for d in diffs]

    rng = random.Random(seed)
    n = len(centered)
    count = 0
    for _ in range(n_samples):
        sample = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(sample) >= abs(observed):
            count += 1
    return count / n_samples


def validate_against_pytrec_eval(run, qrels, k: int = 10) -> Dict[str, float] | None:
    """Optional sanity check. Returns None if pytrec_eval is not installed."""
    try:
        import pytrec_eval
    except ImportError:
        return None
    formatted = {q: {d: 1.0 / (i + 1) for i, d in enumerate(docs)} for q, docs in run.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}", f"recall.{k}", "recip_rank"})
    results = evaluator.evaluate(formatted)
    n = len(results)
    keys = [f"ndcg_cut_{k}", f"recall_{k}", "recip_rank"]
    return {key: sum(r[key] for r in results.values()) / n for key in keys}
