#!/usr/bin/env python
"""Build the blind answer-scoring sheet, and summarise it once you have scored it.

    python scripts/answer_sheet.py                 # write results/answer_scoring.csv
    python scripts/answer_sheet.py --summarize     # after scoring: means per reader

Reads the answers of the two readers,
    results/answers.json                                  (generative, flan-t5-base)
    results/set_answer.mode=extractive/answers.json       (extractive, roberta-base-squad2)
puts all of them in ONE random order without saying which reader wrote which, and adds
what the rubric (docs/ANSWER_RUBRIC.md) needs to score each one: the claim, the gold
abstract, whether it was among the passages the reader got, and the cited passage. The
reader of each item is written to a separate key file used only by --summarize.

An existing sheet that already has scores is never overwritten unless you pass --force.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import load_config  # noqa: E402
from hybridqa.data import build_chunks, load_dataset_for  # noqa: E402
from hybridqa.scoring import CRITERIA, check_row, summarize_scores  # noqa: E402

READERS = {
    "generative": REPO / "results" / "answers.json",
    "extractive": REPO / "results" / "set_answer.mode=extractive" / "answers.json",
}
SHEET = REPO / "results" / "answer_scoring.csv"
KEY = REPO / "results" / "answer_scoring_key.json"


def build(args) -> None:
    if SHEET.exists() and not args.force:
        with open(SHEET, newline="", encoding="utf-8-sig") as fh:
            if any(str(r.get("correctness", "")).strip() for r in csv.DictReader(fh)):
                sys.exit(f"{SHEET} already has scores; not overwriting it (pass --force to replace it)")
    missing = [str(p) for p in READERS.values() if not p.exists()]
    if missing:
        sys.exit("missing answers: " + ", ".join(missing) + "\nrun the two `hybridqa.cli answer` commands first")

    cfg = load_config(args.config)
    corpus, _queries, qrels = load_dataset_for(cfg)
    text = {c.chunk_id: c.indexable for c in build_chunks(corpus, cfg)}

    items = []
    for reader, path in READERS.items():
        for rec in json.loads(path.read_text(encoding="utf-8"))["answers"]:
            gold = sorted(qrels.get(rec["query_id"], {}))
            given_docs = {cid.split("#", 1)[0] for cid in rec.get("passages", [])}
            first_gold = gold[0] if gold else ""
            items.append({
                "reader": reader,
                "query_id": rec["query_id"],
                "claim": rec["query"],
                # the trailing "[n]" would give the generative reader away; the cited
                # passage is shown in its own column for both readers instead
                "answer": re.sub(r"(\s*\[\d+\])+\s*$", "", rec["answer"]),
                "correctness": "", "completeness": "", "faithfulness": "", "citation_accuracy": "", "notes": "",
                "gold_in_passages": "yes" if given_docs & set(gold) else "no",
                "gold_title": (corpus.get(first_gold, {}).get("title") or "").strip(),
                "gold_abstract_start": (corpus.get(first_gold, {}).get("text") or "")[:400],
                "cited_passage": " || ".join(text.get(cid, "")[:500] for cid in rec.get("citations", [])),
            })

    random.Random(cfg.get_path("evaluation.seed", 42)).shuffle(items)
    key = {}
    rows = []
    for i, item in enumerate(items, start=1):
        key[str(i)] = {"reader": item.pop("reader"), "query_id": item["query_id"]}
        rows.append({"item": i, **item})

    with open(SHEET, "w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel reads UTF-8 correctly
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} answers in random order to {SHEET}")
    print(f"the reader of each item is in {KEY.name}: do not open it before you finish scoring")
    print("score with docs/ANSWER_RUBRIC.md, then run with --summarize")


def summarize() -> None:
    if not SHEET.exists() or not KEY.exists():
        sys.exit("no scoring sheet yet; build it first")
    with open(SHEET, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    problems = [p for row in rows for p in check_row(row)]
    if problems:
        sys.exit("fix these first:\n  " + "\n  ".join(problems))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    scored = sum(1 for r in rows if str(r.get("correctness", "")).strip())
    summary = summarize_scores(rows, key)

    auto = {}
    for reader, path in READERS.items():
        if path.exists():
            auto[reader] = json.loads(path.read_text(encoding="utf-8")).get("citation_metrics", {})

    print(f"{scored} of {len(rows)} answers scored (0-2 scale; faithfulness over answers with content)")
    print(f"{'reader':<12}{'n':>4}{'no content':>12}" + "".join(f"{c:>19}" for c in CRITERIA))
    for reader, s in summary.items():
        print(f"{reader:<12}{s['n']:>4}{s['no_content']:>12}"
              + "".join(f"{s[c]:>19}" if s[c] is not None else f"{'-':>19}" for c in CRITERIA))
    out = REPO / "results" / "answer_quality.json"
    out.write_text(json.dumps({"scored": scored, "total": len(rows), "per_reader": summary,
                               "automatic_citation_metrics": auto}, indent=2), encoding="utf-8")
    print(f"written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite a sheet that already has scores")
    ap.add_argument("--summarize", action="store_true", help="average the scores per reader")
    args = ap.parse_args()
    summarize() if args.summarize else build(args)


if __name__ == "__main__":
    main()
