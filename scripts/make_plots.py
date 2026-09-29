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

# Names as they appear in the paper's table
LABELS = {"bm25": "BM25", "dense": "Dense", "fusion": "RRF fusion", "fusion+rerank": "Fusion + rerank"}
# Lines are told apart by dash pattern and marker, not only colour, so the figure still
# works printed in black and white (the colours are just a bonus on screen)
LINE_STYLES = {
    "bm25": dict(color="#7b8794", linestyle=":", marker="s"),
    "dense": dict(color="#52606d", linestyle="-.", marker="^"),
    "fusion": dict(color=INK, linestyle="-", marker="o"),
    "fusion+rerank": dict(color=ACCENT, linestyle="--", marker="D"),
}

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 8,
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
    # two-line labels so four bars fit a 3.5-inch column
    variants = [LABELS.get(v["variant"], v["variant"]).replace(" + ", " +\n").replace("RRF ", "RRF\n")
                for v in data["variants"]]
    ndcg = [v.get("nDCG@10", 0) for v in data["variants"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))  # IEEE column width: prints at 100%
    bars = ax.bar(variants, ndcg, color=GREYS[: len(variants)], width=0.6)
    for bar, val in zip(bars, ndcg):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, max(ndcg) * 1.18 if ndcg else 1)
    ax.set_title("Retrieval ablation", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_ndcg.png")
    plt.close(fig)


def plot_recall_curve(data: dict) -> None:
    ks = [1, 3, 5, 10, 20, 100]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))  # IEEE column width: prints at 100%
    for v in data["variants"]:
        ys = [v.get(f"Recall@{k}") for k in ks]
        if any(y is None for y in ys):
            continue
        style = LINE_STYLES.get(v["variant"], dict(color=INK, linestyle="-", marker="o"))
        ax.plot(ks, ys, markersize=4, linewidth=1.5, label=LABELS.get(v["variant"], v["variant"]),
                **style)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall as a function of cutoff", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recall_curve.png")
    plt.close(fig)


def plot_latency_accuracy(data: dict) -> None:
    """Two panels over reranking depth N: nDCG@10 (left) and latency (right).

    "none" is RRF fusion without the reranker, the baseline every depth must beat.
    The accuracy axis is kept wide on purpose, with BM25 and dense as reference lines:
    a zoomed axis would make a non-significant 0.006 difference look like a big effect.
    """
    sweep_path = RESULTS / "sweep_rerank_depth.json"
    by_name = {v["variant"]: v for v in data["variants"]}
    if not sweep_path.exists() or "fusion" not in by_name:
        print("  skipped latency_accuracy.png: needs sweep_rerank_depth.json and a fusion row "
              "(run_ablation.py --sweep-depth ...)")
        return

    sweep = sorted(json.loads(sweep_path.read_text(encoding="utf-8")),
                   key=lambda r: r["rerank_candidates"])
    labels = ["none"] + [str(r["rerank_candidates"]) for r in sweep]
    xs = list(range(len(labels)))
    ndcg = [by_name["fusion"]["nDCG@10"]] + [r["nDCG@10"] for r in sweep]
    latency = [by_name["fusion"]["latency_ms_per_query"]] + [r["latency_ms_per_query"] for r in sweep]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.5))  # full page width (figure*)

    left.plot(xs, ndcg, color=ACCENT, marker="o", markersize=4, linewidth=1.5)
    refs = [by_name[r]["nDCG@10"] for r in ("dense", "bm25") if r in by_name]
    for ref in ("dense", "bm25"):
        if ref in by_name:
            y = by_name[ref]["nDCG@10"]
            left.axhline(y, color="#7b8794", linestyle=":", linewidth=1)
            left.text(xs[-1], y + 0.004, f"{LABELS[ref]} alone", ha="right", va="bottom",
                      fontsize=7, color="#52606d")
    left.set_ylim(min(ndcg + refs) - 0.04, max(ndcg + refs) + 0.04)
    left.set_ylabel("nDCG@10")
    left.set_title("Accuracy", loc="left", fontweight="bold")

    right.plot(xs, latency, color=INK, marker="s", markersize=4, linewidth=1.5)
    right.set_ylim(0, max(latency) * 1.15)
    right.set_ylabel("latency (ms / query)")
    right.set_title("Cost", loc="left", fontweight="bold")

    for ax in (left, right):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("candidates reranked (N)")
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
