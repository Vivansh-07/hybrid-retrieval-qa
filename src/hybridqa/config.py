"""Config loading. A single YAML file drives every stage so experiments are reproducible."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """Dict with attribute access and dotted-path lookup, so cfg.get_path('dense.top_k') works."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def override(self, dotted: str, value: Any) -> "Config":
        """Return a copy with one dotted key replaced (used by the ablation runner)."""
        new = copy.deepcopy(dict(self))
        node = new
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return Config(new)


def load_config(path: str | Path = "configs/scifact.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve(cfg: Config, key: str) -> Path:
    """Resolve a `paths.*` entry to an absolute directory, creating it if needed."""
    rel = cfg.get_path(f"paths.{key}", key)
    out = Path(rel)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    return out
