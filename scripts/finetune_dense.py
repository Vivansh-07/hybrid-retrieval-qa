#!/usr/bin/env python
"""Fine-tune the dense bi-encoder on SciFact's TRAINING claims (contrastive learning).

    python scripts/finetune_dense.py
    python scripts/finetune_dense.py --epochs 8 --lr 3e-5

Why: the off-the-shelf encoder was trained on general web text, and SciFact is scientific
claims. BEIR ships 809 training claims with their evidence papers; none of them is among
the 300 test claims, so training on them is not tuning on the test set.

What it learns: for a batch of (claim, evidence passage) pairs, pull each claim towards
its own passage and push it away from the other passages in the batch ("in-batch
negatives"). That is a cross-entropy loss over the claim-passage similarity matrix,
where the correct answer for row i is column i.

Protocol:
  1. Hold out part of the training claims as a DEV set. It is never trained on.
  2. After every epoch, retrieve for the dev claims and measure nDCG@10.
  3. Keep the weights of the best dev epoch (early stopping), so the test set never
     influences a training decision.
  4. The test set is only touched afterwards, by run_ablation.py.

Writes the model to data/models/<name>/ (not committed: it is ~90 MB and reproducible)
and the training history to results/finetune_<name>/history.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.config import apply_overrides, load_config, resolve_model  # noqa: E402
from hybridqa.data import build_chunks, chunk_to_doc, load_beir_hf  # noqa: E402
from hybridqa.dense import DenseRetriever  # noqa: E402
from hybridqa.metrics import evaluate_run  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def embed(model, texts):
    """Encode a list of texts WITH gradients (model.encode() switches them off)."""
    features = model.preprocess(texts)
    features = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in features.items()}
    return F.normalize(model(features)["sentence_embedding"], dim=-1)


def contrastive_loss(query_vecs, passage_vecs, scale: float = 20.0):
    """Multiple-negatives ranking loss.

    scores[i][j] = cosine(claim i, passage j) * scale. Passage i is the evidence for
    claim i, every other passage in the batch is a negative, so the target for row i is
    column i. `scale` sharpens the softmax (cosines live in [-1, 1], too flat on their own).
    """
    scores = query_vecs @ passage_vecs.T * scale
    targets = torch.arange(len(scores), device=scores.device)
    return F.cross_entropy(scores, targets)


def best_chunk_pairs(retriever, queries, qrels, chunks_by_doc):
    """One (claim, passage) training pair per judged (claim, paper).

    Relevance is judged per paper, but the index holds 3-sentence chunks. As the positive
    we take the chunk of the evidence paper that the starting model already finds most
    similar to the claim: a weak label, but it needs no manual annotation.
    """
    pairs = []
    query_vecs = retriever.encode([q["text"] for q in queries], is_query=True)
    for q, qvec in zip(queries, query_vecs):
        for doc_id in qrels[q["query_id"]]:
            chunks = chunks_by_doc.get(doc_id, [])
            if not chunks:
                continue
            sims = retriever.encode([c.indexable for c in chunks]) @ qvec
            pairs.append((q["text"], chunks[int(sims.argmax())].indexable))
    return pairs


def dev_scores(retriever, chunks, queries, qrels):
    """Dense-only retrieval for the dev claims with the model as it is right now."""
    retriever.index(chunks)  # re-embeds the whole corpus with the current weights
    results = retriever.search_batch([q["text"] for q in queries])
    run = {q["query_id"]: chunk_to_doc(cid for cid, _ in ranked) for q, ranked in zip(queries, results)}
    metrics = evaluate_run(run, qrels, k_values=[10])
    return metrics["nDCG@10"], metrics["Recall@10"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/scifact.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. dense.model=BAAI/bge-small-en-v1.5 to fine-tune that model")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dev-queries", type=int, default=120, help="training claims held out for dev")
    ap.add_argument("--name", default=None, help="output name (default: <base model>-scifact)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed = cfg.get_path("evaluation.seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    base = cfg.get_path("dense.model")
    name = args.name or base.split("/")[-1] + "-scifact"
    model_dir = REPO / "data" / "models" / name
    out_dir = REPO / "results" / f"finetune_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data: TRAIN split only. The test split is never loaded in this script. ----------
    corpus, queries, qrels = load_beir_hf(cfg.get_path("dataset.hf_id", "BeIR/scifact"), split="train")
    chunks = build_chunks(corpus, cfg)
    chunks_by_doc = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)

    queries = sorted(queries, key=lambda q: q["query_id"])
    random.Random(seed).shuffle(queries)
    dev_q, train_q = queries[: args.dev_queries], queries[args.dev_queries:]
    print(f"[finetune] base model {base}")
    print(f"[finetune] {len(train_q)} training claims, {len(dev_q)} dev claims, {len(chunks)} passages")

    retriever = DenseRetriever(cfg, args.device)
    model = retriever.model  # the SentenceTransformer we are going to update in place
    prefix = cfg.get_path("dense.query_prefix", "")

    pairs = best_chunk_pairs(retriever, train_q, qrels, chunks_by_doc)
    pairs = [(prefix + claim, passage) for claim, passage in pairs]
    print(f"[finetune] {len(pairs)} (claim, evidence passage) training pairs")

    steps_per_epoch = len(pairs) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.1 * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # learning rate: linear warm-up for the first 10% of steps, then linear decay to zero
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / (total_steps - warmup)))

    # ---- epoch 0 = the untouched base model, so the curve shows what training added ------
    ndcg, recall = dev_scores(retriever, chunks, dev_q, qrels)
    history = [{"epoch": 0, "train_loss": None, "dev_nDCG@10": round(ndcg, 4), "dev_Recall@10": round(recall, 4)}]
    print(f"[finetune] epoch 0 (before training): dev nDCG@10 = {ndcg:.4f}  Recall@10 = {recall:.4f}")
    best = {"epoch": 0, "dev_nDCG@10": ndcg}
    model.save(str(model_dir))

    shuffler = random.Random(seed)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffler.shuffle(pairs)
        losses = []
        for step in range(steps_per_epoch):  # the last partial batch is dropped
            batch = pairs[step * args.batch_size : (step + 1) * args.batch_size]
            loss = contrastive_loss(embed(model, [c for c, _ in batch]), embed(model, [p for _, p in batch]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        ndcg, recall = dev_scores(retriever, chunks, dev_q, qrels)  # encode() switches to eval mode
        train_loss = sum(losses) / len(losses)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "dev_nDCG@10": round(ndcg, 4), "dev_Recall@10": round(recall, 4)})
        kept = ""
        if ndcg > best["dev_nDCG@10"]:  # early stopping: keep only the best dev epoch
            best = {"epoch": epoch, "dev_nDCG@10": ndcg}
            model.save(str(model_dir))
            kept = "  <- best so far, saved"
        print(f"[finetune] epoch {epoch}: train loss = {train_loss:.4f}  dev nDCG@10 = {ndcg:.4f}  "
              f"Recall@10 = {recall:.4f}{kept}")

    summary = {
        "base_model": base, "output_model": f"data/models/{name}",
        "train_claims": len(train_q), "dev_claims": len(dev_q), "train_pairs": len(pairs),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": seed,
        "loss": "multiple-negatives ranking (in-batch negatives), scale 20",
        "best_epoch": best["epoch"], "best_dev_nDCG@10": round(best["dev_nDCG@10"], 4),
        "train_seconds": round(time.perf_counter() - t0, 1), "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[finetune] best epoch {best['epoch']} (dev nDCG@10 = {best['dev_nDCG@10']:.4f}) saved to {model_dir}")
    print(f"[finetune] history written to {out_dir / 'history.json'}")
    print("\nNow evaluate on the TEST claims:")
    extra = f' --set "dense.query_prefix={prefix}"' if prefix else ""
    print(f"  python scripts/run_ablation.py --set dense.model=data/models/{name}{extra} "
          f"--out results/dense_{name.replace('-', '_')}")


if __name__ == "__main__":
    main()
