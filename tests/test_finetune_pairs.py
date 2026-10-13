"""finetune_dense.best_chunk_pairs: the cap keeps the most relevant papers; no cap changes nothing."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.data import Chunk  # noqa: E402


def _finetune():
    spec = importlib.util.spec_from_file_location("finetune_dense", REPO / "scripts" / "finetune_dense.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SameVectors:
    """Stands in for the encoder: every text gets the same vector, so the first chunk wins."""

    def encode(self, texts, is_query=False):
        return np.ones((len(texts), 2), dtype="float32")


QUERIES = [{"query_id": "q1", "text": "claim"}]
QRELS = {"q1": {"d1": 1, "d2": 2, "d3": 1, "d4": 2}}
CHUNKS = {d: [Chunk(f"{d}#0", d, "", f"text of {d}")] for d in ("d1", "d2", "d3", "d4")}


def test_no_cap_keeps_every_judged_paper_in_order():
    pairs = _finetune().best_chunk_pairs(_SameVectors(), QUERIES, QRELS, CHUNKS)
    assert [p for _, p in pairs] == ["text of d1", "text of d2", "text of d3", "text of d4"]


def test_cap_keeps_the_most_relevant_first():
    pairs = _finetune().best_chunk_pairs(_SameVectors(), QUERIES, QRELS, CHUNKS, max_per_query=3)
    assert [p for _, p in pairs] == ["text of d2", "text of d4", "text of d1"]   # grade 2, then 1
