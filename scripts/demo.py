"""Interactive demo: one question, four retrieval strategies side by side.

    python scripts/demo.py              # opens http://127.0.0.1:7860 in the browser

Builds the same indexes as the ablation (first launch downloads SciFact and the
models; dense embeddings are then cached in data/indexes/), and for any question
shows what BM25, dense, RRF fusion and fusion+rerank each retrieve, plus a grounded
answer with [n] citations from the full pipeline.

Pick a SciFact test claim from the dropdown and the gold evidence documents are
marked with a check, so you can see *where* each strategy finds or misses them.
The "Results" tab shows results/ablation.csv and the figures once they exist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# hybridqa first: it points the Hugging Face cache at data/hf_cache, and gradio
# imports huggingface_hub, which reads that setting once at import time.
from hybridqa.answer import AnswerGenerator, faithfulness_signals  # noqa: E402
from hybridqa.config import load_config, resolve  # noqa: E402
from hybridqa.pipeline import VARIANTS, RetrievalPipeline  # noqa: E402

os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "data" / "gradio_tmp"))  # not C:\...\Temp

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

TOP_N = 5  # documents shown per variant
LABELS = {
    "bm25": "1 · BM25 (sparse)",
    "dense": "2 · Dense bi-encoder",
    "fusion": "3 · RRF fusion",
    "fusion+rerank": "4 · Fusion + cross-encoder",
}
FIGURES = ["ablation_ndcg.png", "recall_curve.png", "latency_accuracy.png"]


def build_app(cfg_path: str, device: str | None) -> gr.Blocks:
    cfg = load_config(cfg_path)
    pipe = RetrievalPipeline(cfg, device=device).prepare()
    gen = AnswerGenerator(cfg, device=device)
    chunks = {c.chunk_id: c for c in pipe.chunks}
    qid_by_text = {q["text"]: q["query_id"] for q in pipe.queries}

    def render_ranking(variant: str, ranked, gold: set, ms: float) -> str:
        # Evaluation is per document, so the demo collapses chunks to documents too.
        lines = [f"### {LABELS[variant]}", f"*{ms:.0f} ms*", ""]
        seen = set()
        for cid, _score in ranked:
            chunk = chunks[cid]
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            mark = " ✅" if chunk.doc_id in gold else ""
            lines.append(f"**{len(seen)}. {chunk.title}**{mark}  \n"
                         f"<sub>doc {chunk.doc_id} — {chunk.text[:200]}…</sub>\n")
            if len(seen) == TOP_N:
                break
        return "\n".join(lines)

    def render_answer(question: str, ranked) -> str:
        ans = gen.answer("demo", question, ranked, pipe.chunk_texts)
        signals = faithfulness_signals(ans)
        lines = [f"### Answer\n\n{ans.text}", "", "**Passages given to the reader:**"]
        for i, (cid, text) in enumerate(ans.supporting, start=1):
            cited = " ← cited" if cid in ans.cited_chunk_ids else ""
            lines.append(f"- [{i}] doc {chunks[cid].doc_id}{cited}: <sub>{text[:160]}…</sub>")
        lines.append(f"\n<sub>Automatic signals (not a substitute for manual scoring): "
                     f"has_citation={signals['has_citation']:.0f}, "
                     f"lexical_overlap={signals['lexical_overlap']:.2f}, "
                     f"abstained={signals['abstained']:.0f}</sub>")
        return "\n".join(lines)

    def run(question: str, want_answer: bool):
        question = (question or "").strip()
        if not question:
            return [""] * len(VARIANTS) + ["Type a question or pick a test claim.", ""]

        qid = qid_by_text.get(question)
        gold = set(pipe.qrels.get(qid, {})) if qid else set()

        panels, final_ranking = [], []
        for variant in VARIANTS:
            t0 = time.perf_counter()
            ranked = pipe.retrieve(question, variant)
            panels.append(render_ranking(variant, ranked, gold, (time.perf_counter() - t0) * 1000))
            final_ranking = ranked  # the last variant is the full system

        if qid:
            note = f"SciFact test claim **{qid}** — gold evidence doc(s): {', '.join(sorted(gold))} (marked ✅)"
        else:
            note = "Free-text question: no relevance judgements, so nothing is marked ✅."
        answer = render_answer(question, final_ranking) if want_answer else ""
        return panels + [note, answer]

    def load_results():
        csv_path = resolve(cfg, "results_dir") / "ablation.csv"
        table = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(
            {"status": ["No results yet: run scripts/run_ablation.py first."]})
        figures = [str(p) for p in (ROOT / "paper" / "figures" / f for f in FIGURES) if p.exists()]
        return table, figures

    claims = [q["text"] for q in pipe.queries]

    with gr.Blocks(title="Hybrid Retrieval QA") as app:
        gr.Markdown("# Hybrid Sparse–Dense Retrieval with Cross-Encoder Reranking\n"
                    "SciFact (BEIR): 5,183 scientific abstracts. One question, four retrieval "
                    "strategies, same index — only the strategy changes.")
        with gr.Tab("Live demo"):
            with gr.Row():
                claim_picker = gr.Dropdown(claims, label="Pick a SciFact test claim (gold evidence is known)",
                                           scale=3)
                want_answer = gr.Checkbox(value=True, label="Generate cited answer", scale=1)
            question = gr.Textbox(label="…or type any question",
                                  placeholder="Does vitamin D reduce respiratory infections?")
            go = gr.Button("Retrieve", variant="primary")
            note = gr.Markdown()
            with gr.Row():
                panels = [gr.Markdown() for _ in VARIANTS]
            answer = gr.Markdown()

            claim_picker.change(lambda c: c, claim_picker, question)
            outputs = panels + [note, answer]
            go.click(run, [question, want_answer], outputs)
            question.submit(run, [question, want_answer], outputs)

        with gr.Tab("Results"):
            refresh = gr.Button("Reload results")
            table = gr.Dataframe(label="Ablation (results/ablation.csv)")
            gallery = gr.Gallery(label="Figures (paper/figures/)", columns=3, height="auto")
            refresh.click(load_results, None, [table, gallery])
            app.load(load_results, None, [table, gallery])

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/scifact.yaml")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab automatically")
    args = p.parse_args()
    build_app(args.config, args.device).launch(server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
