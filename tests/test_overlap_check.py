"""scripts/overlap_check.py finds the runs of words the full paper shares with another text."""

import importlib.util
import io
import json
import re
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("overlap_check", REPO / "scripts" / "overlap_check.py")
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)


def test_a_shared_run_is_found_whole_and_short_ones_are_ignored():
    source = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    words = "We write that ALPHA beta gamma delta epsilon zeta eta theta iota and then alpha beta gamma".split()
    assert oc.shared_runs(words, source, n=8) == [(3, 12)]  # nine words, case ignored; the last three too short


def test_separate_runs_stay_separate():
    source = "one two three four five six seven eight nine ten eleven twelve".split()
    words = "one two three four x nine ten eleven twelve".split()
    assert oc.shared_runs(words, source, n=4) == [(0, 4), (5, 9)]


def test_tikz_code_is_not_prose(tmp_path):
    # the system diagram's code is shared by both papers, but it is not text a checker reads
    path = tmp_path / "chapter.tex"
    path.write_text("Words before.\n\\begin{tikzpicture}\\node (bm) at (0,0) {BM25 index};\\end{tikzpicture}\n"
                    "Words after.", encoding="utf-8")
    assert oc.document_words(path) == ["Words", "before", "Words", "after"]


def test_every_reference_gets_a_query():
    bib = (REPO / "paper_full" / "references.bib").read_text(encoding="utf-8")
    refs = dict(oc.references())
    assert list(refs) == re.findall(r"^@\w+\{\s*([^,\s]+)", bib, re.M)
    assert refs["robertson2009bm25"] == "DOI:10.1561/1500000019"
    assert refs["nogueira2019passage"] == "arXiv:1901.04085"
    assert refs["lewis2020rag"].startswith("title:Retrieval-Augmented Generation")  # no DOI or arXiv id
    assert refs["scispace"] is None  # a website: no abstract


class _FakeWeb:
    """Stands in for urllib.request.urlopen: records each URL and answers with a fixed reply."""

    def __init__(self, reply=None, status=None):
        self.urls, self.reply, self.status = [], reply, status

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        if self.status:
            raise urllib.error.HTTPError(url, self.status, "error", {}, None)
        return io.BytesIO(json.dumps(self.reply).encode())


def test_abstract_is_fetched_once_then_read_from_the_cache(tmp_path, monkeypatch):
    web = _FakeWeb({"title": "T", "abstract": "An abstract."})
    monkeypatch.setattr(oc, "CACHE", tmp_path)
    monkeypatch.setattr(oc.urllib.request, "urlopen", web)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    assert oc.abstract("k", "DOI:10.1/x") == "An abstract."
    assert oc.abstract("k", "DOI:10.1/x") == "An abstract."
    assert web.urls == [oc.API + "DOI:10.1/x?fields=title,abstract"]


def test_a_title_query_uses_the_best_match(tmp_path, monkeypatch):
    web = _FakeWeb({"data": [{"title": "Some Title", "abstract": "Matched."}]})
    monkeypatch.setattr(oc, "CACHE", tmp_path)
    monkeypatch.setattr(oc.urllib.request, "urlopen", web)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    assert oc.abstract("k", "title:Some Title") == "Matched."
    assert web.urls[0].startswith(oc.API + "search/match?query=Some+Title")


def test_an_unknown_paper_has_no_abstract(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "CACHE", tmp_path)
    monkeypatch.setattr(oc.urllib.request, "urlopen", _FakeWeb(status=404))
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    assert oc.abstract("k", "DOI:10.1/missing") == ""
    assert (tmp_path / "k.txt").read_text(encoding="utf-8") == ""  # not asked again


def test_the_offline_check_lists_every_chapter(capsys):
    assert oc.main([]) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    for name in oc.ORDER:
        assert re.search(rf"^{name}\s+\d+\s+\d+\.\d%", out, re.M), name
