#!/usr/bin/env python
"""Run the four-variant ablation plus the reranking-depth sweep.

    python scripts/run_ablation.py --config configs/scifact.yaml
    python scripts/run_ablation.py --sweep-depth 10 25 50 100

Everything it writes lands in results/ and is what the paper's tables are built from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config, resolve  # noqa: E402
from hybridqa.metrics import paired_bootstrap  # noqa: E402
from hybridqa.pipeline import VARIANTS, RetrievalPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--sweep-depth", nargs="*", type=int, default=[],
                    help="rerank candidate depths to sweep, e.g. 10 25 50 100")
    ap.add_argument("--sweep-fusion-k", nargs="*", type=int, default=[],
                    help="RRF k values to sweep, e.g. 10 30 60 100")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t0 = time.perf_counter()
    pipe = RetrievalPipeline(cfg, device=args.device).prepare()

    results = pipe.run_all(args.variants)
    out_dir = pipe.save_results(results)

    # significance: is fusion+rerank really better than the strongest single retriever?
    primary_k = cfg.get_path("evaluation.primary_k", 10)
    key = f"nDCG@{primary_k}"
    sig = {}
    if "fusion+rerank" in results:
        target = {q: m[key] for q, m in results["fusion+rerank"].per_query.items()}
        for base in ("bm25", "dense", "fusion"):
            if base in results:
                baseline = {q: m[key] for q, m in results[base].per_query.items()}
                sig[f"fusion+rerank vs {base}"] = {
                    "delta": round(
                        sum(target.values()) / len(target) - sum(baseline.values()) / len(baseline), 4
                    ),
                    "p_value": paired_bootstrap(baseline, target),
                }
    if sig:
        (out_dir / "significance.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
        print("\n=== Paired bootstrap on " + key + " ===")
        for name, s in sig.items():
            print(f"  {name}: delta={s['delta']:+.4f}  p={s['p_value']:.4f}")

    # rerank depth sweep: the accuracy-vs-latency curve the paper needs
    if args.sweep_depth:
        sweep = []
        for depth in args.sweep_depth:
            pipe.reranker.candidates = depth
            res = pipe.run_variant("fusion+rerank")
            row = {"rerank_candidates": depth, **res.summary()}
            sweep.append(row)
            print(f"  depth={depth:>4}  {key}={res.metrics[key]:.4f}  "
                  f"{res.latency_ms_per_query:.1f} ms/query")
        (out_dir / "sweep_rerank_depth.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    # RRF k sweep: shows the fusion result is not an artefact of one hyperparameter
    if args.sweep_fusion_k:
        sweep = []
        original = pipe.cfg
        for kval in args.sweep_fusion_k:
            pipe.cfg = original.override("fusion.k", kval)
            res = pipe.run_variant("fusion")
            sweep.append({"rrf_k": kval, **res.summary()})
            print(f"  rrf_k={kval:>4}  {key}={res.metrics[key]:.4f}")
        pipe.cfg = original
        (out_dir / "sweep_rrf_k.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f}s")
    print(f"artifacts in {resolve(cfg, 'results_dir')}")


if __name__ == "__main__":
    main()
