#!/usr/bin/env python
"""Choose the BM25 weight in fusion on held-out DEV claims, never on test claims.

    python scripts/tune_fusion.py
    python scripts/tune_fusion.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact

RRF weights both retrievers equally by default. When one retriever is much stronger
(e.g. the fine-tuned dense encoder), the weaker one's vote can pull good results down.
This script tries several weights for BM25 (dense stays 1.0) on the 120 dev claims of the
SciFact TRAINING split (the same ones finetune_dense.py held out, so the fine-tuned model
never trained on them), picks the best by nDCG@10, and prints the one command that then
evaluates that weight on the test claims. A weight of 0 means dense alone.

Writes results/fusion_weights_<dense model>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import apply_overrides, load_config  # noqa: E402
from hybridqa.data import chunk_to_doc, dev_split  # noqa: E402
from hybridqa.fusion import fuse  # noqa: E402
from hybridqa.metrics import evaluate_run  # noqa: E402
from hybridqa.pipeline import RetrievalPipeline  # noqa: E402

GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dev-queries", type=int, default=120, help="must match finetune_dense.py")
    args = ap.parse_args()

    base = apply_overrides(load_config(args.config), args.overrides)
    cfg = apply_overrides(base, ["dataset.split=train"])  # TRAINING claims only
    pipe = RetrievalPipeline(cfg).prepare()
    _, dev = dev_split(pipe.queries, args.dev_queries, cfg.get_path("evaluation.seed", 42))
    qrels = {q["query_id"]: pipe.qrels[q["query_id"]] for q in dev}

    # retrieve once; only the fusion weights change between grid points
    lists = {q["query_id"]: (pipe.bm25.search(q["text"]), pipe.dense.search(q["text"])) for q in dev}

    grid = []
    print(f"\nnDCG@10 on {len(dev)} dev claims (training split), dense weight 1.0")
    for w in GRID:
        wcfg = cfg.override("fusion.weights", [w, 1.0])
        run = {qid: chunk_to_doc(cid for cid, _ in fuse(list(pair), wcfg)) for qid, pair in lists.items()}
        m = evaluate_run(run, qrels, k_values=[10])
        grid.append({"bm25_weight": w, "nDCG@10": round(m["nDCG@10"], 4), "Recall@10": round(m["Recall@10"], 4)})
        print(f"  BM25 weight {w:>4}:  nDCG@10 = {m['nDCG@10']:.4f}   Recall@10 = {m['Recall@10']:.4f}")

    best = max(grid, key=lambda g: (g["nDCG@10"], -abs(g["bm25_weight"] - 1.0)))  # tie: closer to equal
    model = cfg.get_path("dense.model")
    name = model.replace("/", "__").replace("\\", "__")
    out = REPO / "results" / f"fusion_weights_{name.split('__')[-1]}.json"
    out.write_text(json.dumps({"dense_model": model, "dev_claims": len(dev), "grid": grid, "chosen": best},
                              indent=2), encoding="utf-8")
    print(f"\nchosen on dev: BM25 weight {best['bm25_weight']} (nDCG@10 {best['nDCG@10']:.4f}); written to {out}")
    extra = "".join(f' --set "{o}"' for o in args.overrides)
    print("evaluate it ONCE on the test claims:")
    print(f'  python scripts/run_ablation.py --variants fusion fusion+rerank{extra} '
          f'--set "fusion.weights=[{best["bm25_weight"]}, 1.0]" --out results/<name>')


if __name__ == "__main__":
    main()
