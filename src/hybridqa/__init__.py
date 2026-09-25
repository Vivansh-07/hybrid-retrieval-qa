"""Hybrid sparse-dense retrieval with cross-encoder reranking for domain-specific QA."""

import os
from pathlib import Path

__version__ = "0.1.0"

# Keep every download (SciFact, models) inside the project's data/ folder instead of the
# default C:\Users\<you>\.cache\huggingface. Must run before huggingface_hub is imported,
# which is why it lives here: any `import hybridqa...` triggers it first.
# setdefault: an HF_HOME you set yourself still wins.
_DATA = Path(__file__).resolve().parents[2] / "data"
os.environ.setdefault("HF_HOME", str(_DATA / "hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_DATA / "mpl_cache"))  # matplotlib's font cache
