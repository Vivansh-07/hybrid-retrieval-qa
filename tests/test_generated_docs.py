"""The two generated documents in the repository match the sources they are built from.

docs/Problem_Statement.pdf and paper/Literature_Survey.docx are attached to the release
(.github/workflows/release.yml) but committed rather than built there, so an edit to
their source that is not followed by a rebuild would ship an old version. Each test
builds the document again and compares its text with the committed file.
"""

import importlib.util
import re
from pathlib import Path

import pytest
from docx import Document

REPO = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docx_text(doc) -> list[str]:
    body = [p.text for p in doc.paragraphs]
    cells = [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    return body + cells


def test_survey_docx_matches_the_markdown():
    fresh = _docx_text(_script("make_survey_docx").build())
    committed = _docx_text(Document(str(REPO / "paper" / "Literature_Survey.docx")))
    assert fresh == committed, (
        "paper/Literature_Survey.docx is out of date: run python scripts/make_survey_docx.py")


def _pdf_text(path: Path) -> list[str]:
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(str(path))
    try:
        return [re.sub(r"\s+", " ", page.get_textpage().get_text_range()).strip() for page in pdf]
    finally:
        pdf.close()


def test_problem_statement_pdf_matches_its_script(tmp_path):
    fresh = tmp_path / "Problem_Statement.pdf"
    _script("make_problem_statement").build(fresh)
    assert _pdf_text(fresh) == _pdf_text(REPO / "docs" / "Problem_Statement.pdf"), (
        "docs/Problem_Statement.pdf is out of date: run python scripts/make_problem_statement.py")
