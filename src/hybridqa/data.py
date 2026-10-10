"""Corpus / query / qrels loading and chunking for BEIR-style datasets.

Data model used everywhere downstream:
    Chunk  -> {"chunk_id", "doc_id", "text", "title"}
    Query  -> {"query_id", "text"}
    Qrels  -> {query_id: {doc_id: relevance_int}}

Relevance is judged at DOCUMENT level in BEIR, so a chunk counts as relevant
if its parent doc_id is relevant. `chunk_to_doc` handles that mapping.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def indexable(self) -> str:
        """Title is prepended: in scientific abstracts it carries most of the topical signal."""
        return f"{self.title}. {self.text}".strip() if self.title else self.text


def split_sentences(text: str) -> List[str]:
    """Cheap regex sentence splitter. Deliberately dependency-free and deterministic.

    Good enough for abstracts; swap in nltk.sent_tokenize if you need more accuracy
    (and say so in the paper's Limitations section).
    """
    text = " ".join(text.split())
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    window: int = 3,
    stride: int = 2,
    min_chars: int = 120,
    keep_tail: bool = False,
) -> List[Chunk]:
    """Sliding sentence window. Overlap keeps evidence that straddles a boundary retrievable.

    A window shorter than `min_chars` is too short to stand alone. With `keep_tail` its
    sentences are folded into the previous chunk and the window moves on. Without it
    (the default, which every reported result uses) the document stops there, so the
    sentences after a short window are never indexed: on SciFact 414 sentences of 42
    abstracts (issue #45). The default changes after the assessment, with all results rerun.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    if window <= 0 or len(sents) <= window:
        return [Chunk(f"{doc_id}#0", doc_id, title, " ".join(sents))]

    chunks: List[Chunk] = []
    covered = 0  # sentences[:covered] are in some chunk
    step = max(1, stride)
    for i in range(0, len(sents), step):
        end = min(i + window, len(sents))
        piece = " ".join(sents[i:end])
        if len(piece) < min_chars and chunks:
            if not keep_tail:
                break
            if end > covered:
                last = chunks[-1]
                chunks[-1] = Chunk(last.chunk_id, doc_id, title, " ".join([last.text, *sents[covered:end]]))
        else:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, piece))
        covered = max(covered, end)
        if end >= len(sents):
            break
    return chunks


def build_chunks(corpus: Dict[str, dict], cfg) -> List[Chunk]:
    """Apply the configured chunking strategy to a {doc_id: {title, text}} corpus."""
    strategy = cfg.get_path("chunking.strategy", "sentence_window")
    window = cfg.get_path("chunking.window_sentences", 3)
    stride = cfg.get_path("chunking.stride_sentences", 2)
    min_chars = cfg.get_path("chunking.min_chars", 120)
    keep_tail = cfg.get_path("chunking.keep_tail", False)

    chunks: List[Chunk] = []
    for doc_id, doc in corpus.items():
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        if strategy == "none":
            chunks.append(Chunk(f"{doc_id}#0", doc_id, title, text))
        else:
            chunks.extend(chunk_document(doc_id, title, text, window, stride, min_chars, keep_tail))
    return chunks


def chunk_to_doc(chunk_ids: Iterable[str]) -> List[str]:
    """Map chunk ids back to parent doc ids, preserving order and dropping duplicates."""
    seen, out = set(), []
    for cid in chunk_ids:
        did = cid.split("#", 1)[0]
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


# --------------------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------------------

def load_beir_hf(hf_id: str = "BeIR/scifact", split: str = "test", cache_dir: Optional[str] = None):
    """Load a BEIR dataset from the HuggingFace Hub.

    Returns (corpus, queries, qrels). Requires `datasets`. The qrels live in a
    separate repo (`<hf_id>-qrels`) — that is BEIR's layout on the Hub, not a bug.
    """
    from datasets import load_dataset

    corpus_ds = load_dataset(hf_id, "corpus", split="corpus", cache_dir=cache_dir)
    queries_ds = load_dataset(hf_id, "queries", split="queries", cache_dir=cache_dir)
    qrels_ds = load_dataset(f"{hf_id}-qrels", split=split, cache_dir=cache_dir)

    corpus = {
        str(row["_id"]): {"title": row.get("title", ""), "text": row.get("text", "")}
        for row in corpus_ds
    }

    qrels: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid, did, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if score > 0:
            qrels.setdefault(qid, {})[did] = score

    # keep only queries that have judgments in this split
    queries = [
        {"query_id": str(row["_id"]), "text": row["text"]}
        for row in queries_ds
        if str(row["_id"]) in qrels
    ]
    return corpus, queries, qrels


def load_local_jsonl(data_dir: str | Path):
    """Fallback loader for a BEIR folder on disk (corpus.jsonl, queries.jsonl, qrels/test.tsv)."""
    data_dir = Path(data_dir)
    corpus: Dict[str, dict] = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}

    qrels: Dict[str, Dict[str, int]] = {}
    qrels_file = next((data_dir / "qrels").glob("*.tsv"))
    with open(qrels_file, encoding="utf-8") as fh:
        header = fh.readline()  # query-id corpus-id score
        assert "query" in header.lower()
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")[:3]
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    queries = []
    with open(data_dir / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries.append({"query_id": str(row["_id"]), "text": row["text"]})
    return corpus, queries, qrels


def load_dataset_for(cfg, cache_dir: Optional[str] = None):
    """Config-driven entry point used by the pipeline and scripts."""
    local = cfg.get_path("dataset.local_dir")
    if local:
        corpus, queries, qrels = load_local_jsonl(local)
    else:
        corpus, queries, qrels = load_beir_hf(
            cfg.get_path("dataset.hf_id", "BeIR/scifact"),
            cfg.get_path("dataset.split", "test"),
            cache_dir=cache_dir,
        )

    max_docs = cfg.get_path("dataset.max_docs")
    if max_docs:
        # keep every judged document, then pad up to max_docs — otherwise a truncated
        # corpus silently makes recall look impossible
        judged = {d for rels in qrels.values() for d in rels}
        kept = {d: corpus[d] for d in judged if d in corpus}
        for did, doc in corpus.items():
            if len(kept) >= max_docs:
                break
            kept.setdefault(did, doc)
        corpus = kept
    return corpus, queries, qrels


def dev_split(queries: List[dict], n_dev: int, seed: int = 42):
    """Split TRAINING claims into (train, dev) the same way every time.

    Sorted by id first, then shuffled with a fixed seed, so the fine-tuning script and
    the fusion-weight tuning use exactly the same 120 held-out dev claims.
    """
    ordered = sorted(queries, key=lambda q: q["query_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[n_dev:], ordered[:n_dev]
