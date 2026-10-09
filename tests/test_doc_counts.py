"""The test count quoted in the docs matches the real one.

The run guide's "Expect: N passed" is shown live in class, so a stale number looks like
a failure. This collects the tests (without running them) and checks every quote.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (file, text with {n} for the count)
QUOTES = [
    ("README.md", "# {n} tests"),
    ("README.md", "{n} tests check the metrics"),
    ("docs/RUN_AND_PRESENT.md", "**Expect:** `{n} passed`"),
    ("docs/DEPLOY.md", "Expect `{n} passed`"),
    ("docs/VIVA_PREP.md", "(1) {n} unit tests"),
]


def collected_tests() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return int(re.search(r"(\d+) tests? collected", out).group(1))


def test_documented_test_counts_are_current():
    n = collected_tests()
    stale = [f"{name}: '{text.format(n=n)}' not found"
             for name, text in QUOTES
             if text.format(n=n) not in (REPO / name).read_text(encoding="utf-8")]
    assert not stale, f"the suite has {n} tests; update: " + "; ".join(stale)
