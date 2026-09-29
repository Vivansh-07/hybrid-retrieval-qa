#!/usr/bin/env python
"""Run the four-variant ablation plus the reranking-depth sweep.

    python scripts/run_ablation.py --config configs/scifact.yaml
    python scripts/run_ablation.py --sweep-depth 10 25 50 100

    # one-change experiments, without editing the YAML:
    python scripts/run_ablation.py --set chunking.strategy=none
    python scripts/run_ablation.py --variants fusion --set fusion.method=nsf

The main run writes to results/, which is what the paper's tables are built from.
A run with --set writes to its own folder (results/set_<key>=<value>/) unless --out
says otherwise, so side experiments never overwrite the main results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import apply_overrides, load_config, output_dir  # noqa: E402
from hybridqa.pipeline import VARIANTS, RetrievalPipeline  # noqa: E402
from hybridqa.significance import save_significance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--sweep-depth", nargs="*", type=int, default=[],
                    help="rerank candidate depths to sweep, e.g. 10 25 50 100")
    ap.add_argument("--sweep-fusion-k", nargs="*", type=int, default=[],
                    help="RRF k values to sweep, e.g. 10 30 60 100")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="change one config value for this run, e.g. chunking.strategy=none "
                         "(repeatable)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: results/, or results/set_... with --set)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    out_dir = output_dir(cfg, args.out, args.overrides)
    t0 = time.perf_counter()
    pipe = RetrievalPipeline(cfg, device=args.device).prepare()

    results = pipe.run_all(args.variants)
    out_dir = pipe.save_results(results, out_dir)

    # significance: every pair of variants, so each claim in the paper has a p-value
    primary_k = cfg.get_path("evaluation.primary_k", 10)
    key = f"nDCG@{primary_k}"
    if len(results) > 1:
        save_significance({name: res.per_query for name, res in results.items()}, out_dir,
                          key=key, seed=cfg.get_path("evaluation.seed", 42))

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
    print(f"artifacts in {out_dir}")


if __name__ == "__main__":
    main()
