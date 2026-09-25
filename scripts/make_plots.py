#!/usr/bin/env python
"""Turn results/*.json into the paper's figures.

    python scripts/make_plots.py

Writes paper/figures/{ablation_ndcg.png, recall_curve.png, latency_accuracy.png}.
Figures are greyscale-safe and use a single accent colour, because the printed
copy your examiner reads may well be black and white.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "data" / "mpl_cache"))  # keep font cache off C:

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = REPO / "results"
FIGDIR = REPO / "paper" / "figures"

INK = "#1f2933"
ACCENT = "#2563eb"
GREYS = ["#cbd2d9", "#9aa5b1", "#616e7c", ACCENT]

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9aa5b1",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#e4e7eb",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def load_metrics() -> dict:
    path = RESULTS / "metrics.json"
    if not path.exists():
        sys.exit(f"no results yet — run scripts/run_ablation.py first (looked for {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_ablation(data: dict) -> None:
    variants = [v["variant"] for v in data["variants"]]
    ndcg = [v.get("nDCG@10", 0) for v in data["variants"]]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bars = ax.bar(variants, ndcg, color=GREYS[: len(variants)], width=0.6)
    for bar, val in zip(bars, ndcg):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, max(ndcg) * 1.18 if ndcg else 1)
    ax.set_title("Retrieval ablation", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_ndcg.png")
    plt.close(fig)


def plot_recall_curve(data: dict) -> None:
    ks = [1, 3, 5, 10, 20, 100]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, v in enumerate(data["variants"]):
        ys = [v.get(f"Recall@{k}") for k in ks]
        if any(y is None for y in ys):
            continue
        ax.plot(ks, ys, marker="o", markersize=3.5, linewidth=1.6,
                color=GREYS[i % len(GREYS)], label=v["variant"])
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall as a function of cutoff", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recall_curve.png")
    plt.close(fig)


def plot_latency_accuracy(data: dict) -> None:
    sweep_path = RESULTS / "sweep_rerank_depth.json"
    fig, ax = plt.subplots(figsize=(5.2, 3.0))

    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        xs = [r["latency_ms_per_query"] for r in sweep]
        ys = [r["nDCG@10"] for r in sweep]
        ax.plot(xs, ys, marker="o", color=ACCENT, linewidth=1.6, markersize=4)
        for r in sweep:
            ax.annotate(f"N={r['rerank_candidates']}",
                        (r["latency_ms_per_query"], r["nDCG@10"]),
                        textcoords="offset points", xytext=(5, -9), fontsize=7)
        ax.set_title("Reranking depth: accuracy vs latency", loc="left", fontweight="bold")
    else:
        for i, v in enumerate(data["variants"]):
            ax.scatter(v["latency_ms_per_query"], v.get("nDCG@10", 0),
                       color=GREYS[i % len(GREYS)], s=42, label=v["variant"])
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Accuracy vs latency by variant", loc="left", fontweight="bold")

    ax.set_xlabel("latency (ms / query)")
    ax.set_ylabel("nDCG@10")
    fig.tight_layout()
    fig.savefig(FIGDIR / "latency_accuracy.png")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    plot_ablation(data)
    plot_recall_curve(data)
    plot_latency_accuracy(data)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
