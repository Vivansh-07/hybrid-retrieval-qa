"""Answer generation over retrieved passages, with citations back to source documents.

Two modes:
  extractive  -- a span-selection QA head (roberta-base-squad2) picks a span from the
                 single best passage. Cheap, and the citation is exact by construction.
  generative  -- flan-t5 conditions on the top-n passages, each tagged [1]..[n], and is
                 instructed to cite. Fluent, but must be checked for faithfulness, which
                 is why `faithfulness_signals` below is reported alongside the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class Answer:
    query_id: str
    query: str
    text: str
    cited_chunk_ids: List[str] = field(default_factory=list)
    supporting: List[Tuple[str, str]] = field(default_factory=list)  # (chunk_id, passage)

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.text,
            "citations": self.cited_chunk_ids,
            "supporting_doc_ids": sorted({c.split("#", 1)[0] for c in self.cited_chunk_ids}),
            # every passage the reader was given, in order: [1] is the first. Needed to
            # judge an abstention ("was the evidence even there?") when scoring by hand.
            "passages": [cid for cid, _ in self.supporting],
        }


PROMPT = (
    "Answer the question using only the numbered passages below. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. "
    "If the passages do not contain the answer, reply exactly: Not enough evidence.\n\n"
    "{passages}\n\nQuestion: {question}\nAnswer:"
)


class AnswerGenerator:
    def __init__(self, cfg, device: str | None = None):
        self.mode = cfg.get_path("answer.mode", "generative")
        self.n_passages = cfg.get_path("answer.n_passages", 5)
        self.max_new_tokens = cfg.get_path("answer.max_new_tokens", 128)
        self.gen_model_name = cfg.get_path("answer.generative_model", "google/flan-t5-base")
        self.ext_model_name = cfg.get_path("answer.extractive_model", "deepset/roberta-base-squad2")
        self.device = device
        self._tokenizer = None
        self._model = None

    # Models are called directly rather than through transformers.pipeline(): the
    # "text2text-generation" and "question-answering" pipelines were removed in
    # transformers 5, and the explicit version shows exactly what the reader does.
    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if self.mode == "extractive":
                name, model_cls = self.ext_model_name, AutoModelForQuestionAnswering
            else:
                name, model_cls = self.gen_model_name, AutoModelForSeq2SeqLM
            self._tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = model_cls.from_pretrained(name).to(device).eval()
        return self._tokenizer, self._model

    def _generate(self, prompt: str) -> str:
        """Greedy decoding with flan-t5: deterministic, so answers are reproducible."""
        import torch

        tok, model = self._load()
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return tok.decode(ids[0], skip_special_tokens=True)

    def _extract_span(self, question: str, passage: str, max_answer_tokens: int = 30) -> str:
        """Pick the highest-scoring (start, end) span inside the passage, end >= start."""
        import torch

        tok, model = self._load()
        inputs = tok(question, passage, return_tensors="pt", truncation="only_second",
                     max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        start_logits, end_logits = out.start_logits[0], out.end_logits[0]

        # only tokens that belong to the passage (sequence id 1) may be part of the answer
        in_passage = torch.tensor([s == 1 for s in inputs.sequence_ids(0)], device=model.device)
        start_logits = start_logits.masked_fill(~in_passage, float("-inf"))
        end_logits = end_logits.masked_fill(~in_passage, float("-inf"))

        scores = start_logits[:, None] + end_logits[None, :]           # score[s, e]
        n = scores.size(0)
        valid = torch.ones(n, n, device=model.device).triu().tril(max_answer_tokens - 1).bool()
        scores = scores.masked_fill(~valid, float("-inf"))
        start, end = divmod(int(scores.argmax()), n)
        return tok.decode(inputs["input_ids"][0][start : end + 1], skip_special_tokens=True).strip()

    def answer(
        self,
        query_id: str,
        query: str,
        ranked: Sequence[Tuple[str, float]],
        chunk_texts: Dict[str, str],
    ) -> Answer:
        selected = [(cid, chunk_texts[cid]) for cid, _ in ranked[: self.n_passages] if cid in chunk_texts]
        if not selected:
            return Answer(query_id, query, "Not enough evidence.", [], [])

        if self.mode == "extractive":
            cid, passage = selected[0]
            return Answer(query_id, query, self._extract_span(query, passage), [cid], selected)

        numbered = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(selected, start=1))
        prompt = PROMPT.format(passages=numbered, question=query)
        out = self._generate(prompt)

        cited_idx = {int(m) for m in re.findall(r"\[(\d+)\]", out) if 1 <= int(m) <= len(selected)}
        cited = [selected[i - 1][0] for i in sorted(cited_idx)] or [selected[0][0]]
        return Answer(query_id, query, out.strip(), cited, selected)


def faithfulness_signals(answer: Answer) -> Dict[str, float]:
    """Cheap automatic proxies to report before the manual annotation pass.

    These are NOT a substitute for reading 50 answers by hand — say so in the paper.
      has_citation      did the model cite anything at all
      lexical_overlap   fraction of answer content words that appear in a cited passage
                        (low overlap is a hallucination smell, not proof of one)
      abstained         did it correctly use the escape hatch
    """
    words = {w for w in re.findall(r"[a-z]{4,}", answer.text.lower())}
    cited_text = " ".join(t for cid, t in answer.supporting if cid in answer.cited_chunk_ids).lower()
    support_words = set(re.findall(r"[a-z]{4,}", cited_text))
    overlap = len(words & support_words) / len(words) if words else 0.0
    return {
        "has_citation": float(bool(answer.cited_chunk_ids)),
        "lexical_overlap": overlap,
        "abstained": float("not enough evidence" in answer.text.lower()),
    }


def citation_accuracy(answers: Sequence[Answer], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Fraction of cited documents that are actually gold evidence, and recall over gold docs."""
    precisions, recalls = [], []
    for a in answers:
        gold = {d for d, r in qrels.get(a.query_id, {}).items() if r > 0}
        if not gold:
            continue
        cited_docs = {c.split("#", 1)[0] for c in a.cited_chunk_ids}
        if cited_docs:
            precisions.append(len(cited_docs & gold) / len(cited_docs))
        recalls.append(len(cited_docs & gold) / len(gold))
    return {
        "citation_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "citation_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "n_evaluated": float(len(recalls)),
    }
