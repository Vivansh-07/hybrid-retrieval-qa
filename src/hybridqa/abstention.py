"""Could a retrieval signal tell the reader to abstain? Scoring helpers for scripts/abstention.py.

A claim is a *hit* when a gold abstract is among the passages the reader is given, and a
*miss* otherwise. A signal (the top cross-encoder score, the top dense similarity) is
useful if it is higher for hits; the reader would then abstain below a threshold. Rows
are dicts with a boolean "gold_in_passages" and one float per signal.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a random hit > score of a random miss), ties count half (Mann-Whitney)."""
    if not pos or not neg:
        raise ValueError("AUC needs at least one hit and one miss")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _split(rows: List[dict], signal: str):
    hits = [r[signal] for r in rows if r["gold_in_passages"]]
    misses = [r[signal] for r in rows if not r["gold_in_passages"]]
    return hits, misses


def best_threshold(rows: List[dict], signal: str) -> float:
    """Abstain when the signal is below t; pick t maximising caught misses - refused hits (rates).

    Candidates are the observed values; on a tie the lowest threshold wins.
    """
    hits, misses = _split(rows, signal)
    if not hits or not misses:
        raise ValueError("choosing a threshold needs at least one hit and one miss")
    candidates = sorted({r[signal] for r in rows})
    return max(candidates, key=lambda t: sum(m < t for m in misses) / len(misses)
               - sum(h < t for h in hits) / len(hits))


def apply_threshold(rows: List[dict], signal: str, t: float) -> Dict[str, float]:
    """How many misses the threshold catches and how many hits it refuses."""
    hits, misses = _split(rows, signal)
    return {
        "threshold": round(t, 4),
        "misses_caught": sum(m < t for m in misses), "misses": len(misses),
        "hits_refused": sum(h < t for h in hits), "hits": len(hits),
    }
