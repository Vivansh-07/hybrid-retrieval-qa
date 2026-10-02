"""Repository hygiene: no stray control characters in tracked text files.

Twice an escaped backslash (as in data\\models\\all-...) turned into an invisible control
character in the docs. This keeps that from reaching the paper or the guide again.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT = (".py", ".md", ".tex", ".bib", ".yaml", ".yml", ".txt", ".toml", ".json", ".csv", ".bat", ".sh")


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / f for f in out.split() if f.endswith(TEXT)]


def test_no_control_characters_in_tracked_text_files():
    bad = {}
    for path in tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [i for i, ch in enumerate(text) if ord(ch) < 32 and ch not in "\n\r\t"]
        if hits:
            bad[str(path.relative_to(REPO))] = hits[:3]
    assert not bad, f"control characters (offsets) in: {bad}"
