"""Dense bi-encoder retrieval over a FAISS flat inner-product index.

Query and passage are embedded independently, so the whole corpus can be encoded
once offline and searched in milliseconds. The cost is that the two never see each
other: matching is a single dot product between two fixed vectors, which is why a
cross-encoder reranker (see rerank.py) still has something to add.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .config import resolve_model
from .data import Chunk


class DenseRetriever:
    def __init__(self, cfg, device: str | None = None):
        self.model_name = cfg.get_path("dense.model", "sentence-transformers/all-MiniLM-L6-v2")
        self.query_prefix = cfg.get_path("dense.query_prefix", "")
        self.batch_size = cfg.get_path("dense.batch_size", 128)
        self.max_seq_length = cfg.get_path("dense.max_seq_length", 256)
        self.normalize = cfg.get_path("dense.normalize", True)
        self.top_k = cfg.get_path("dense.top_k", 100)
        self.device = device
        self.chunk_ids: List[str] = []
        self._model = None
        self._index = None
        self.embeddings: np.ndarray | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # resolve_model: a fine-tuned model saved under data/models/ is loaded from disk
            self._model = SentenceTransformer(resolve_model(self.model_name), device=self.device)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 1000,
        )
        return vecs.astype("float32")

    def index(self, chunks: Sequence[Chunk], cache: str | Path | None = None) -> "DenseRetriever":
        import faiss

        self.chunk_ids = [c.chunk_id for c in chunks]
        cache = Path(cache) if cache else None

        if cache and cache.exists():
            self.embeddings = np.load(cache)
            if len(self.embeddings) != len(chunks):
                raise ValueError(
                    f"cached embeddings ({len(self.embeddings)}) do not match the corpus "
                    f"({len(chunks)}) — delete {cache} and re-index"
                )
        else:
            self.embeddings = self.encode([c.indexable for c in chunks])
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.embeddings)

        dim = self.embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # exact search: 5K x 384 is trivial
        self._index.add(self.embeddings)
        return self

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: Sequence[str], top_k: int | None = None) -> List[List[Tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("call .index(chunks) before searching")
        top_k = min(top_k or self.top_k, len(self.chunk_ids))
        qvecs = self.encode(queries, is_query=True)
        scores, idxs = self._index.search(qvecs, top_k)
        return [
            [(self.chunk_ids[i], float(s)) for i, s in zip(row_i, row_s) if i != -1]
            for row_i, row_s in zip(idxs, scores)
        ]
