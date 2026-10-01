#!/usr/bin/env python
"""Which retriever helps which kind of claim? Compare the variants inside claim subsets.

    python scripts/query_subsets.py
    python scripts/query_subsets.py --results results/dense_minilm_finetuned

Splits the 300 test claims four ways (see src/hybridqa/subsets.py for the definitions):
contains a number or not; rare vs common vocabulary; low vs high word overlap with the
evidence abstract; short vs long. Inside every group it reports the mean nDCG@10 of each
variant and paired bootstrap tests between them. Reads the saved per-claim results
(<results>/runs/*.json), so it takes seconds and retrieves nothing.

Writes <results>/query_subsets.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.data import load_dataset_for  # noqa: E402
from hybridqa.significance import format_p, load_runs  # noqa: E402
from hybridqa.sparse import tokenize  # noqa: E402
from hybridqa.subsets import boolean_split, claim_properties, compare_in_groups, median_split  # noqa: E402

PAIRS = [("bm25", "dense"), ("bm25", "fusion"), ("dense", "fusion"), ("fusion", "fusion+rerank")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--results", default="results", help="folder containing runs/ (default: results)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = Path(args.results)
    results = results if results.is_absolute() else REPO / results
    runs = load_runs(results)
    if not {"bm25", "dense"} <= set(runs):
        sys.exit(f"need at least the bm25 and dense run files in {results / 'runs'}")
    per_query = {v: saved["per_query"] for v, saved in runs.items()}

    corpus, queries, qrels = load_dataset_for(cfg)
    tok = lambda text: tokenize(text, cfg.get_path("sparse.lowercase", True),  # noqa: E731
                                cfg.get_path("sparse.remove_stopwords", True), cfg.get_path("sparse.stem", True))
    props = claim_properties(queries, corpus, qrels, tok)

    splits = {"number": (boolean_split(props, "has_number", "contains a number", "no number"), None)}
    splits["vocabulary"] = median_split(props, "rarest_df", "rare term", "common terms only")
    splits["overlap"] = median_split(props, "overlap", "low overlap (paraphrased)", "high overlap")
    splits["length"] = median_split(props, "length", "short", "long")

    key = f"nDCG@{cfg.get_path('evaluation.primary_k', 10)}"
    seed = cfg.get_path("evaluation.seed", 42)
    variants = [v for v in ("bm25", "dense", "fusion", "fusion+rerank") if v in per_query]
    payload = {"metric": key, "results": str(results), "splits": {}}

    print(f"\n{key} by claim subset   ({results})\n")
    print(f"{'split':<11}{'group':<27}{'n':>4}  " + "".join(f"{v:>14}" for v in variants)
          + "   dense-bm25 (p)     fusion-dense (p)")
    for split, (groups, cut) in splits.items():
        report = compare_in_groups(per_query, groups, PAIRS, key=key, seed=seed)
        payload["splits"][split] = {"median_cut": cut, "groups": report}
        for name, r in report.items():
            c = r["comparisons"]
            def cmp(name_):
                s = c.get(name_)
                return f"{s['delta']:+.3f} ({format_p(s['p_value'])})" if s else "-"
            print(f"{split:<11}{name:<27}{r['n']:>4}  " + "".join(f"{r['mean'].get(v, float('nan')):>14.4f}" for v in variants)
                  + f"   {cmp('dense vs bm25'):<18} {cmp('fusion vs dense'):<18}")
        if cut is not None:
            print(f"{'':<11}(split at the median: {cut})")

    payload["properties"] = props
    (results / "query_subsets.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {results / 'query_subsets.json'}")


if __name__ == "__main__":
    main()
