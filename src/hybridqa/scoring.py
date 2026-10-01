"""Checking and summarising the hand scores of generated answers (docs/ANSWER_RUBRIC.md).

The scores themselves are given by a person. This module only validates what was typed
and averages it per reader, after joining the blind sheet with its hidden key.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

CRITERIA = ["correctness", "completeness", "faithfulness", "citation_accuracy"]
REQUIRED = ["correctness", "completeness", "citation_accuracy"]  # faithfulness may be blank (no content)


def check_row(row: dict) -> List[str]:
    """Problems with one scored row; empty if it is fine or not scored yet."""
    values = {c: str(row.get(c, "")).strip() for c in CRITERIA}
    if not any(values.values()):
        return []  # not scored yet
    problems = []
    for c in REQUIRED:
        if values[c] not in {"0", "1", "2"}:
            problems.append(f"item {row.get('item')}: {c} must be 0, 1 or 2 (got {values[c]!r})")
    if values["faithfulness"] not in {"", "0", "1", "2"}:
        problems.append(f"item {row.get('item')}: faithfulness must be 0, 1, 2 or blank")
    if values["faithfulness"] == "" and "no_content" not in str(row.get("notes", "")):
        problems.append(f"item {row.get('item')}: blank faithfulness is only for answers noted 'no_content'")
    return problems


def summarize_scores(rows: Sequence[dict], key: Dict[str, dict]) -> dict:
    """Mean of each criterion per reader. Faithfulness is averaged over answers with content."""
    per_reader: Dict[str, dict] = {}
    for row in rows:
        if not str(row.get("correctness", "")).strip():
            continue  # not scored yet
        reader = key[str(row["item"])]["reader"]
        acc = per_reader.setdefault(reader, {"n": 0, "no_content": 0, **{c: [] for c in CRITERIA}})
        acc["n"] += 1
        acc["no_content"] += "no_content" in str(row.get("notes", ""))
        for c in CRITERIA:
            value = str(row.get(c, "")).strip()
            if value:
                acc[c].append(int(value))
    return {
        reader: {
            "n": acc["n"],
            "no_content": acc["no_content"],
            **{c: round(sum(acc[c]) / len(acc[c]), 3) if acc[c] else None for c in CRITERIA},
        }
        for reader, acc in per_reader.items()
    }
