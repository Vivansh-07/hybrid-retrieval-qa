#!/usr/bin/env python
"""Does the full paper repeat long runs of words from another text? A plagiarism pre-check.

    python scripts/overlap_check.py                 against the project's other documents
    python scripts/overlap_check.py --abstracts     also against the abstracts of the references

Plagiarism checkers flag runs of words that also occur in a source. This finds every run of at
least 8 words (--min-words) that a chapter of paper_full/ shares with
- the project's documents that are submitted or published on their own: the short paper
  (paper/main.tex), the literature survey (docs/LITERATURE_SURVEY.md) and the problem statement
  (docs/Problem_Statement.pdf);
- with --abstracts, the abstract of every reference in paper_full/references.bib that has a DOI
  or an arXiv id (else, for a published work, its title), from the Semantic Scholar API. This needs
  internet; the abstracts are saved in data/abstracts/, so a second run is offline.
It prints, per chapter, the share of its words that lie inside a shared run, then the runs
themselves. Quotation marks and citations are not looked at: a run may be a cited quotation, a
name or a title, so read each one. Numbers, LaTeX commands and TikZ diagrams are not words here.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from rewrite_progress import ORDER, SECTIONS, prose  # noqa: E402

MIN_WORDS = 8   # shorter shared runs are mostly common phrasing, and checkers usually ignore them
DOCUMENTS = {"short paper": "paper/main.tex", "survey": "docs/LITERATURE_SURVEY.md",
             "problem statement": "docs/Problem_Statement.pdf"}
BIB = REPO / "paper_full" / "references.bib"
CACHE = REPO / "data" / "abstracts"
API = "https://api.semanticscholar.org/graph/v1/paper/"
TIKZ = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)


def plain_words(text: str) -> list:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def document_words(path: Path) -> list:
    """The words of a .tex (prose only), .md or .pdf file."""
    if path.suffix == ".tex":   # a diagram's code shared by both papers is not prose
        return prose(TIKZ.sub(" ", path.read_text(encoding="utf-8")))
    if path.suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return plain_words(" ".join(page.get_textpage().get_text_range() for page in pdf))
        finally:
            pdf.close()
    return plain_words(path.read_text(encoding="utf-8"))


def shared_runs(words: list, source: list, n: int = MIN_WORDS) -> list:
    """(start, end) of every maximal run of at least n words that also occurs in source.

    Case-insensitive. A run is the union of overlapping n-word windows found in the source.
    """
    low = [w.lower() for w in words]
    grams = {tuple(x.lower() for x in source[i:i + n]) for i in range(len(source) - n + 1)}
    runs = []
    for i in range(len(low) - n + 1):
        if tuple(low[i:i + n]) in grams:
            if runs and i <= runs[-1][1]:
                runs[-1] = (runs[-1][0], i + n)
            else:
                runs.append((i, i + n))
    return runs


def references() -> list:
    """(key, Semantic Scholar query) for every entry of references.bib, in order.

    The query is an arXiv id or a DOI, else the title of a published work, else None (a website
    such as a literature assistant's home page has no abstract to compare with).
    """
    out = []
    for entry in re.split(r"\n(?=@)", BIB.read_text(encoding="utf-8")):
        head = re.match(r"@\w+\{\s*([^,\s]+)", entry)
        if not head:
            continue
        field = lambda name: re.search(rf"^\s*{name}\s*=\s*\{{(.+)\}},?\s*$", entry, re.M)  # noqa: E731
        doi, arxiv, title = field("doi"), field("eprint"), field("title")
        published = any(field(f) for f in ("journaltitle", "booktitle", "institution", "publisher"))
        query = (f"arXiv:{arxiv.group(1)}" if arxiv else f"DOI:{doi.group(1)}" if doi   # arXiv's own DOIs resolve less often
                 else f"title:{re.sub(r'[{}]', '', title.group(1))}" if title and published else None)
        out.append((head.group(1), query))
    return out


def abstract(key: str, query: str) -> str:
    """The reference's abstract ("" if Semantic Scholar has none), cached in data/abstracts/."""
    cached = CACHE / f"{key}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    if query.startswith("title:"):   # best title match
        url = API + "search/match?" + urllib.parse.urlencode({"query": query[6:], "fields": "title,abstract"})
    else:
        url = API + urllib.parse.quote(query, safe=":/") + "?fields=title,abstract"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                found = json.load(resp)
            text = ((found.get("data") or [{}])[0] if query.startswith("title:") else found).get("abstract") or ""
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                text = ""
                break
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))   # rate limit without an API key: wait and retry
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    time.sleep(1.1)   # about one request per second keeps under the public rate limit
    return text


def sources(with_abstracts: bool) -> dict:
    found = {name: document_words(REPO / rel) for name, rel in DOCUMENTS.items()}
    if with_abstracts:
        missing = []
        for key, query in references():
            text = abstract(key, query) if query else ""
            if text:
                found[key] = plain_words(text)
            else:
                missing.append(key)
        if missing:
            print(f"No abstract to compare for {len(missing)} references: {', '.join(missing)}\n")
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abstracts", action="store_true", help="also compare with the references' abstracts (internet)")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS, help=f"shortest run reported (default {MIN_WORDS})")
    args = ap.parse_args(argv)

    found = sources(args.abstracts)
    print(f"{'chapter':<27}{'words':>6}  {'in shared runs':>14}  longest  sources")
    passages = []
    for name in ORDER:
        words = document_words(SECTIONS / f"{name}.tex")
        covered, longest, hits = set(), 0, []
        for source, other in found.items():
            runs = shared_runs(words, other, args.min_words)
            for start, end in runs:
                covered.update(range(start, end))
                longest = max(longest, end - start)
                passages.append((end - start, name, source, " ".join(words[start:end])))
            if runs:
                hits.append(source)
        share = len(covered) / len(words) if words else 0
        print(f"{name:<27}{len(words):>6}  {share:>13.1%}  {longest:>7}  {', '.join(hits) or '-'}")

    print(f"\nShared runs of {args.min_words} or more words, longest first:")
    for length, name, source, text in sorted(passages, reverse=True):
        print(f"  {length:>3}  {name} / {source}: {text}".encode("ascii", "replace").decode())
    if not passages:
        print("  none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
