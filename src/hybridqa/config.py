"""Config loading. A single YAML file drives every stage so experiments are reproducible."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

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


_MISSING = object()


def apply_overrides(cfg: Config, assignments: Sequence[str] | None) -> Config:
    """Apply command-line overrides such as ``chunking.strategy=none``.

    Lets a run change one setting without editing the YAML (the full config used is
    saved with the results, so the change is still on record). A setting that is
    currently a string is replaced by the text exactly as typed (so a model name, or a
    query prefix ending in ": ", needs no quoting tricks). Anything else is parsed as
    YAML: ``60`` becomes an int, ``true`` a bool, ``[1, 0.5]`` a list. The key must
    already exist, so a typo fails loudly instead of silently running the default
    configuration.
    """
    for item in assignments or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"override must look like key=value, got {item!r}")
        if cfg.get_path(key, _MISSING) is _MISSING:
            section = cfg.get_path(key.rsplit(".", 1)[0], {}) if "." in key else cfg
            options = ", ".join(sorted(section)) if isinstance(section, dict) else "?"
            raise KeyError(f"unknown config key {key!r} (keys here: {options})")
        value = raw if isinstance(cfg.get_path(key), str) else yaml.safe_load(raw)
        cfg = cfg.override(key, value)
    return cfg


def resolve(cfg: Config, key: str) -> Path:
    """Resolve a `paths.*` entry to an absolute directory, creating it if needed."""
    rel = cfg.get_path(f"paths.{key}", key)
    out = Path(rel)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def output_dir(cfg: Config, out: str | Path | None = None,
               overrides: Sequence[str] | None = None) -> Path:
    """Where a run writes its results.

    `out` if given; otherwise results/ for a run with the default config, and a named
    subfolder (results/set_<key>=<value>/) for a run with --set overrides, so a side
    experiment can never overwrite the main results the paper is built from.
    """
    if out:
        path = Path(out)
        path = path if path.is_absolute() else REPO_ROOT / path
    elif overrides:
        name = "__".join(o.replace(" ", "") for o in overrides)
        safe = "".join(c if c.isalnum() or c in "._=-" else "-" for c in name)
        path = resolve(cfg, "results_dir") / f"set_{safe}"
    else:
        path = resolve(cfg, "results_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path
