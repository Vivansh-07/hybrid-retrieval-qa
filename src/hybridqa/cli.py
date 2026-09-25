"""Command-line entry point.

    python -m hybridqa.cli ablation --config configs/scifact.yaml
    python -m hybridqa.cli query "Does vitamin D reduce respiratory infections?"
    python -m hybridqa.cli answer --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybridqa.config import load_config, resolve  # noqa: E402
from hybridqa.pipeline import VARIANTS, RetrievalPipeline  # noqa: E402


def cmd_ablation(args) -> None:
    cfg = load_config(args.config)
    pipe = RetrievalPipeline(cfg, device=args.device).prepare()
    variants = args.variants or list(VARIANTS)
    results = pipe.run_all(variants)
    out = pipe.save_results(results)

    print("\n=== Ablation ===")
    header = ["variant", "nDCG@10", "Recall@10", "Recall@100", "MRR@10", "latency_ms_per_query"]
    print(" | ".join(f"{h:>20}" for h in header))
    for res in results.values():
        s = res.summary()
        print(" | ".join(f"{str(s.get(h, '-')):>20}" for h in header))
    print(f"\nwritten to {out}")


def cmd_query(args) -> None:
    cfg = load_config(args.config)
    pipe = RetrievalPipeline(cfg, device=args.device).prepare()
    ranked = pipe.retrieve(args.text, args.variant)[: args.top_k]

    print(f"\nQuery: {args.text}\nVariant: {args.variant}\n")
    for i, (cid, score) in enumerate(ranked, 1):
        print(f"[{i}] {cid}  score={score:.4f}\n    {pipe.chunk_texts[cid][:300]}...\n")

    if args.answer:
        from hybridqa.answer import AnswerGenerator, faithfulness_signals

        gen = AnswerGenerator(cfg, device=args.device)
        ans = gen.answer("adhoc", args.text, ranked, pipe.chunk_texts)
        print("Answer:", ans.text)
        print("Citations:", ans.cited_chunk_ids)
        print("Signals:", faithfulness_signals(ans))


def cmd_answer(args) -> None:
    """Generate answers for the evaluation queries and score citation accuracy."""
    from hybridqa.answer import AnswerGenerator, citation_accuracy, faithfulness_signals

    cfg = load_config(args.config)
    pipe = RetrievalPipeline(cfg, device=args.device).prepare()
    gen = AnswerGenerator(cfg, device=args.device)

    queries = pipe.queries[: args.limit] if args.limit else pipe.queries
    answers, records = [], []
    for q in queries:
        ranked = pipe.retrieve(q["text"], args.variant)
        ans = gen.answer(q["query_id"], q["text"], ranked, pipe.chunk_texts)
        answers.append(ans)
        records.append({**ans.as_dict(), "signals": faithfulness_signals(ans)})
        print(f"- {q['query_id']}: {ans.text[:120]}")

    out = resolve(cfg, "results_dir") / "answers.json"
    out.write_text(
        json.dumps(
            {"citation_metrics": citation_accuracy(answers, pipe.qrels), "answers": records},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ncitation metrics: {citation_accuracy(answers, pipe.qrels)}")
    print(f"written to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybridqa", description=__doc__)
    p.add_argument("--config", default="configs/scifact.yaml")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ablation", help="run the full retrieval ablation and write results/")
    a.add_argument("--variants", nargs="*", choices=list(VARIANTS))
    a.set_defaults(func=cmd_ablation)

    q = sub.add_parser("query", help="retrieve for one ad-hoc question")
    q.add_argument("text")
    q.add_argument("--variant", default="fusion+rerank", choices=list(VARIANTS))
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--answer", action="store_true", help="also generate a cited answer")
    q.set_defaults(func=cmd_query)

    ans = sub.add_parser("answer", help="generate cited answers for the eval queries")
    ans.add_argument("--variant", default="fusion+rerank", choices=list(VARIANTS))
    ans.add_argument("--limit", type=int, default=50)
    ans.set_defaults(func=cmd_answer)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
