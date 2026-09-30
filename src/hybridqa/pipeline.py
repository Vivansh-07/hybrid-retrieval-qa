"""Pipeline orchestration and the four ablation variants.

    bm25          sparse only
    dense         bi-encoder only
    fusion        RRF(bm25, dense)
    fusion+rerank RRF(bm25, dense) -> cross-encoder      <- the full system

Every variant shares one corpus, one chunking, one set of indexes, so the only
thing that differs between rows of the results table is the retrieval strategy.
That is what makes the ablation an ablation rather than four unrelated systems.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from .config import Config, resolve
from .data import Chunk, build_chunks, chunk_to_doc, load_dataset_for
from .dense import DenseRetriever
from .fusion import fuse
from .metrics import evaluate_run, per_query_metrics
from .rerank import CrossEncoderReranker
from .sparse import BM25Retriever

VARIANTS = ("bm25", "dense", "fusion", "fusion+rerank")


@dataclass
class VariantResult:
    name: str
    metrics: Dict[str, float]
    latency_ms_per_query: float
    run: Dict[str, List[str]] = field(default_factory=dict)
    per_query: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "variant": self.name,
            "latency_ms_per_query": round(self.latency_ms_per_query, 2),
            **{k: round(v, 4) for k, v in self.metrics.items()},
        }


class RetrievalPipeline:
    """Builds the indexes once, then serves any variant."""

    def __init__(self, cfg: Config, device: str | None = None, verbose: bool = True):
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.chunks: List[Chunk] = []
        self.chunk_texts: Dict[str, str] = {}
        self.queries: List[dict] = []
        self.qrels: Dict[str, Dict[str, int]] = {}
        self.bm25: BM25Retriever | None = None
        self.dense: DenseRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.build_stats: Dict[str, float] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}", flush=True)

    # -- setup -------------------------------------------------------------------------
    def prepare(self) -> "RetrievalPipeline":
        t0 = time.perf_counter()
        corpus, self.queries, self.qrels = load_dataset_for(self.cfg)
        self._log(f"loaded {len(corpus)} docs, {len(self.queries)} judged queries")

        self.chunks = build_chunks(corpus, self.cfg)
        self.chunk_texts = {c.chunk_id: c.indexable for c in self.chunks}
        self._log(f"chunked into {len(self.chunks)} passages "
                  f"({len(self.chunks) / max(1, len(corpus)):.2f} per doc)")
        self.build_stats["load_and_chunk_s"] = time.perf_counter() - t0

        t = time.perf_counter()
        self.bm25 = BM25Retriever(self.cfg).index(self.chunks)
        self.build_stats["bm25_index_s"] = time.perf_counter() - t
        self._log(f"BM25 index built in {self.build_stats['bm25_index_s']:.1f}s")

        t = time.perf_counter()
        cache = self.embedding_cache_path()
        from_cache = cache.exists()
        self.dense = DenseRetriever(self.cfg, self.device).index(self.chunks, cache=cache)
        self.build_stats["dense_index_s"] = time.perf_counter() - t
        # a cached run only loads vectors, so its time is NOT the index build cost —
        # record which one it was before quoting dense_index_s in the paper
        self.build_stats["dense_from_cache"] = float(from_cache)
        self.build_stats["embedding_mb"] = (
            self.dense.embeddings.nbytes / 1e6 if self.dense.embeddings is not None else 0.0
        )
        how = "loaded from cache" if from_cache else "built"
        self._log(f"dense index {how} in {self.build_stats['dense_index_s']:.1f}s "
                  f"({self.build_stats['embedding_mb']:.1f} MB)")

        self.reranker = CrossEncoderReranker(self.cfg, self.device)
        return self

    def embedding_cache_path(self) -> Path:
        """Cache file for the passage embeddings: one per (dataset, chunking, encoder).

        The vectors depend on which passages exist and how they were encoded, so all of
        that is hashed into the file name. Changing the chunking or the model then
        re-embeds, instead of loading vectors that belong to a different set of passages.
        """
        keyed = {k: self.cfg.get_path(k) for k in (
            "dataset.name", "dataset.max_docs", "chunking",
            "dense.model", "dense.max_seq_length", "dense.normalize",
        )}
        digest = hashlib.sha1(json.dumps(keyed, sort_keys=True, default=str).encode()).hexdigest()[:10]
        model = self.cfg.get_path("dense.model", "model").replace("/", "__")
        return resolve(self.cfg, "index_dir") / f"{model}__{digest}.npy"

    # -- retrieval ---------------------------------------------------------------------
    def retrieve(self, query: str, variant: str = "fusion+rerank") -> List[Tuple[str, float]]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

        if variant == "bm25":
            return self.bm25.search(query)
        if variant == "dense":
            return self.dense.search(query)

        fused = fuse([self.bm25.search(query), self.dense.search(query)], self.cfg)
        if variant == "fusion":
            return fused
        # Return the whole list (reranked head + original tail), not just rerank.top_k:
        # the other variants are evaluated on ~100 chunks, so cutting this one to 10
        # would make its Recall@20/@100 incomparable.
        return self.reranker.rerank(query, fused, self.chunk_texts, top_k=len(fused))

    def run_variant(self, variant: str, k_values: Sequence[int] | None = None) -> VariantResult:
        k_values = k_values or self.cfg.get_path("evaluation.k_values", [1, 3, 5, 10, 20, 100])
        run: Dict[str, List[str]] = {}
        latencies: List[float] = []

        # One untimed query first. The models load lazily on first use (and download on
        # a first run); without this, that one-time cost is counted as query latency.
        if self.queries:
            self.retrieve(self.queries[0]["text"], variant)

        for q in tqdm(self.queries, desc=variant, disable=not self.verbose):
            t0 = time.perf_counter()
            ranked = self.retrieve(q["text"], variant)
            latencies.append((time.perf_counter() - t0) * 1000)
            # BEIR judges documents, so collapse chunks to their parent doc ids
            run[q["query_id"]] = chunk_to_doc(cid for cid, _ in ranked)

        metrics = evaluate_run(run, self.qrels, k_values)
        primary_k = self.cfg.get_path("evaluation.primary_k", 10)
        return VariantResult(
            name=variant,
            metrics=metrics,
            latency_ms_per_query=sum(latencies) / len(latencies),
            run=run,
            per_query=per_query_metrics(run, self.qrels, primary_k),
        )

    def run_all(self, variants: Sequence[str] = VARIANTS) -> Dict[str, VariantResult]:
        return {v: self.run_variant(v) for v in variants}

    # -- persistence -------------------------------------------------------------------
    def save_results(self, results: Dict[str, VariantResult], out_dir: str | Path | None = None) -> Path:
        out = Path(out_dir) if out_dir else resolve(self.cfg, "results_dir")
        out.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": dict(self.cfg),
            "build_stats": {k: round(v, 3) for k, v in self.build_stats.items()},
            "corpus": {"n_chunks": len(self.chunks), "n_queries": len(self.queries)},
            "variants": [r.summary() for r in results.values()],
        }
        (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        runs_dir = out / "runs"
        runs_dir.mkdir(exist_ok=True)
        for name, res in results.items():
            safe = name.replace("+", "_")
            (runs_dir / f"{safe}.json").write_text(
                json.dumps({"run": res.run, "per_query": res.per_query}, indent=2), encoding="utf-8"
            )

        # flat CSV for the paper's main table
        rows = [r.summary() for r in results.values()]
        if rows:
            cols = list(rows[0].keys())
            lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
            (out / "ablation.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        self._log(f"results written to {out}")
        return out
