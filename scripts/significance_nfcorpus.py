#!/usr/bin/env python
"""Paired bootstrap between configurations on NFCorpus, the full paper's second dataset.

    python scripts/significance_nfcorpus.py

Compares the improved retriever (fine-tuned encoder + weighted fusion) with the standard
pipeline and its parts, per query on nDCG@10, with the project's own paired bootstrap
(10,000 resamples, seed 42). Reads the per-query run files of results/nfcorpus*/ and writes
results/nfcorpus_comparisons.json. A comparison whose runs are missing is skipped.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hybridqa.metrics import paired_bootstrap  # noqa: E402
from hybridqa.significance import format_p  # noqa: E402


def per_query(folder, variant):
    path = REPO / "results" / folder / "runs" / f"{variant.replace('+', '_')}.json"
    return {q: m["nDCG@10"] for q, m in json.loads(path.read_text(encoding="utf-8"))["per_query"].items()}


PAIRS = [  # (name, (folder, variant) of A, (folder, variant) of B): is B better than A?
    ("improved vs standard pipeline (fusion + rerank)",
     ("nfcorpus", "fusion+rerank"), ("nfcorpus_finetuned__fusion_weighted", "fusion")),
    ("improved vs standard fusion (w = 1, pretrained)",
     ("nfcorpus", "fusion"), ("nfcorpus_finetuned__fusion_weighted", "fusion")),
    ("fine-tuned dense vs pretrained dense",
     ("nfcorpus", "dense"), ("nfcorpus_finetuned__fusion_weighted", "dense")),
    ("improved vs fine-tuned dense alone",
     ("nfcorpus_finetuned__fusion_weighted", "dense"), ("nfcorpus_finetuned__fusion_weighted", "fusion")),
    ("SciFact encoder, not retrained (w = 0.3) vs standard fusion",
     ("nfcorpus", "fusion"), ("nfcorpus_scifact_encoder__fusion_weighted", "fusion")),
    ("improved + rerank vs improved",
     ("nfcorpus_finetuned__fusion_weighted", "fusion"), ("nfcorpus_finetuned__fusion_weighted__rerank", "fusion+rerank")),
    ("improved + rerank vs standard pipeline",
     ("nfcorpus", "fusion+rerank"), ("nfcorpus_finetuned__fusion_weighted__rerank", "fusion+rerank")),
]

out = {"metric": "nDCG@10", "test": "two-sided paired bootstrap", "n_samples": 10000, "seed": 42, "comparisons": {}}
for name, (fa, va), (fb, vb) in PAIRS:
    try:
        a, b = per_query(fa, va), per_query(fb, vb)
    except FileNotFoundError:
        print(f"{name}: run missing, skipped")
        continue
    qids = sorted(set(a) & set(b))
    mean_a = sum(a[q] for q in qids) / len(qids)
    mean_b = sum(b[q] for q in qids) / len(qids)
    p = paired_bootstrap(a, b, 10000, 42)
    out["comparisons"][name] = {
        "A": f"{fa}:{va}", "B": f"{fb}:{vb}", "A_mean": round(mean_a, 4), "B_mean": round(mean_b, 4),
        "delta": round(mean_b - mean_a, 4), "p_value": p, "queries": len(qids),
        "better": sum(b[q] > a[q] for q in qids), "worse": sum(b[q] < a[q] for q in qids),
    }
    print(f"{name:<62} {mean_a:.4f} -> {mean_b:.4f}  delta={mean_b - mean_a:+.4f}  p={format_p(p)}")
(REPO / "results" / "nfcorpus_comparisons.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
