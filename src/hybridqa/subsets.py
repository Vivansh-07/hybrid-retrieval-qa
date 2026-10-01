"""Which retriever helps which kind of claim? Query-subset analysis.

The ablation gives one average per variant. Here the claims are split into groups by a
property that can be computed from the text, and the variants are compared inside each
group. If BM25 and dense really fail on different claims, the split should show where:
BM25 should do relatively better on claims full of exact terms, dense on claims phrased
differently from their evidence.

Properties (all from the claim text and the corpus; `overlap` also uses the gold abstract,
so it is for analysis only, never available to a retriever):
  has_number   the claim contains a digit: numbers, doses, gene/protein codes like IL-6
  rarest_df    in how many abstracts the claim's rarest matching term occurs (lower = rarer)
  overlap      share of the claim's terms that also occur in its evidence abstract
  length       number of content terms in the claim
Terms are produced by the BM25 tokenizer (lower-cased, stopwords removed, stemmed), so
"overlap" means exactly the word matching BM25 can use.
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from .significance import compare_runs

Groups = Dict[str, List[str]]  # group name -> query ids


def claim_properties(
    queries: Sequence[dict],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    tokenize: Callable[[str], List[str]],
) -> Dict[str, dict]:
    doc_terms = {d: set(tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")) for d, doc in corpus.items()}
    df = Counter(t for terms in doc_terms.values() for t in terms)

    props = {}
    for q in queries:
        terms = set(tokenize(q["text"]))
        in_corpus = [df[t] for t in terms if df[t] > 0]
        gold = [d for d in qrels.get(q["query_id"], {}) if d in doc_terms]
        overlap = max((len(terms & doc_terms[d]) / len(terms) for d in gold), default=0.0) if terms else 0.0
        props[q["query_id"]] = {
            "has_number": any(ch.isdigit() for ch in q["text"]),
            "rarest_df": min(in_corpus) if in_corpus else 0,
            "overlap": round(overlap, 4),
            "length": len(terms),
        }
    return props


def median_split(props: Dict[str, dict], key: str, low_name: str, high_name: str) -> Tuple[Groups, float]:
    """Two groups of about equal size: at or below the median, and above it."""
    cut = median(p[key] for p in props.values())
    groups = {low_name: [q for q, p in props.items() if p[key] <= cut],
              high_name: [q for q, p in props.items() if p[key] > cut]}
    return groups, cut


def boolean_split(props: Dict[str, dict], key: str, true_name: str, false_name: str) -> Groups:
    return {true_name: [q for q, p in props.items() if p[key]],
            false_name: [q for q, p in props.items() if not p[key]]}


def compare_in_groups(
    per_query: Dict[str, Dict[str, Dict[str, float]]],
    groups: Groups,
    pairs: Iterable[Tuple[str, str]],
    key: str = "nDCG@10",
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, dict]:
    """Mean of every variant and paired bootstrap for every pair, separately inside each group."""
    pairs = list(pairs)
    report = {}
    for name, qids in groups.items():
        ids = set(qids)
        sub = {v: {q: m for q, m in pq.items() if q in ids} for v, pq in per_query.items()}
        report[name] = {
            "n": len(ids),
            "mean": {v: round(sum(m[key] for m in pq.values()) / len(pq), 4) for v, pq in sub.items() if pq},
            "comparisons": compare_runs(sub, pairs, key, n_samples, seed),
        }
    return report
