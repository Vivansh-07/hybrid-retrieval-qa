"""BM25 sparse retrieval.

BM25 scores a query-document pair as a sum over query terms of
    IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
so it rewards exact term overlap and rare terms — exactly the identifiers, gene
names and numeric codes that dense encoders blur away. That is the half of the
hybrid this module contributes.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .data import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

# Small, explicit stopword list — no NLTK download needed at run time.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "he", "her", "his", "how", "i", "in", "is", "it", "its", "of", "on", "or", "she",
    "that", "the", "their", "there", "these", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}


def _porter(token: str) -> str:
    """Lazy Porter stemmer, falling back to identity if NLTK is unavailable."""
    global _STEMMER
    try:
        _STEMMER
    except NameError:
        try:
            from nltk.stem import PorterStemmer

            _STEMMER = PorterStemmer()
        except Exception:  # pragma: no cover - optional dependency
            _STEMMER = None
    return _STEMMER.stem(token) if _STEMMER is not None else token


def tokenize(text: str, lowercase: bool = True, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    tokens = _TOKEN.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_porter(t) for t in tokens]
    return tokens


class BM25Retriever:
    """Wraps rank_bm25 with the project's chunk ids and config."""

    def __init__(self, cfg):
        self.k1 = cfg.get_path("sparse.k1", 0.9)
        self.b = cfg.get_path("sparse.b", 0.4)
        self.lowercase = cfg.get_path("sparse.lowercase", True)
        self.remove_stopwords = cfg.get_path("sparse.remove_stopwords", True)
        self.stem = cfg.get_path("sparse.stem", True)
        self.top_k = cfg.get_path("sparse.top_k", 100)
        self.chunk_ids: List[str] = []
        self._bm25: BM25Okapi | None = None

    def _tok(self, text: str) -> List[str]:
        return tokenize(text, self.lowercase, self.remove_stopwords, self.stem)

    def index(self, chunks: Sequence[Chunk]) -> "BM25Retriever":
        self.chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [self._tok(c.indexable) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("call .index(chunks) before .search()")
        top_k = top_k or self.top_k
        scores = self._bm25.get_scores(self._tok(query))
        # stable: equal scores keep index order, exactly as the earlier Python sort did
        ranked = np.argsort(-np.asarray(scores), kind="stable")[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]

    def search_batch(self, queries: Sequence[str], top_k: int | None = None) -> List[List[Tuple[str, float]]]:
        return [self.search(q, top_k) for q in queries]
