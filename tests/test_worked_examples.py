"""The worked examples in the full paper's Methodology (paper_full/sections/03_methodology.tex).

Each example is recomputed here, with the project's own code where it has a function for it,
and the paper must show the result as the test rounds it. A slip in the arithmetic, or a
rewrite that changes a number, fails here instead of in front of the examiner.
"""

import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridqa.abstention import apply_threshold, auc, best_threshold  # noqa: E402

METHOD = (REPO / "paper_full" / "sections" / "03_methodology.tex").read_text(encoding="utf-8")
# without comments, and with LaTeX's thin spaces and line breaks reduced to single spaces
TEXT = re.sub(r"\s+", " ", "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in METHOD.splitlines()))


def _shown(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _assert_in_paper(*snippets: str) -> None:
    for s in snippets:
        assert s in TEXT, f"the Methodology should show '{s}'"


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def test_idf_example():
    n_passages = 21015
    rare = math.log((n_passages - 10 + 0.5) / (10 + 0.5))
    common = math.log((n_passages - 2000 + 0.5) / (2000 + 0.5))
    assert 3 * common < rare  # "one shared rare term outweighs three shared common ones"
    _assert_in_paper(f"= {_shown(rare, 2)}$", f"= {_shown(common, 2)}$")


def test_length_normalisation_example():
    k1, b = 0.9, 0.4
    denominator = 1 + k1 * (1 - b + b * 2)
    _assert_in_paper(f"= 1.9 / {_shown(denominator, 2)} = {_shown((k1 + 1) / denominator, 2)}$")


def test_cosine_example():
    uq, up = (3, 4), (4, 3)
    vq = [x / math.hypot(*uq) for x in uq]
    vp = [x / math.hypot(*up) for x in up]
    assert vq == [0.6, 0.8] and vp == [0.8, 0.6]
    _assert_in_paper(f"= {_shown(sum(a * b for a, b in zip(vq, vp)), 2)}$")


def test_positive_passage_example():
    similarities = [0.41, 0.63, 0.52]
    assert max(range(3), key=similarities.__getitem__) == 1  # the second passage
    _assert_in_paper("0.41, 0.63 and 0.52", "the second becomes")


@pytest.mark.parametrize("scale", [20, 1])
def test_loss_example(scale):
    own, other = scale * 0.6, scale * 0.4
    p = math.exp(own) / (math.exp(own) + math.exp(other))
    assert abs(p - _sigmoid(own - other)) < 1e-12
    shown_p = _shown(p, 3)
    if scale == 20:
        _assert_in_paper(f"= {shown_p}$", f"-\\ln {shown_p} = {_shown(-math.log(p), 3)}$")
    else:
        _assert_in_paper(f"= {shown_p}$", f"a loss of {_shown(-math.log(p), 3)}")


def test_sigmoid_keeps_the_order():
    high, low = _sigmoid(5.36), _sigmoid(-2.30)
    assert high > low
    _assert_in_paper(f"= {_shown(high, 3)}$", f"= {_shown(low, 3)}$")


def test_attribution_tie_example():
    shared = {1: 1, 2: 4, 3: 4, 4: 0, 5: 1}
    best = max(shared.values())
    assert min(i for i, n in shared.items() if n == best) == 2  # ties go to the higher rank
    _assert_in_paper("$c(y) = 2$")


def test_bootstrap_example():
    x = [0.2, 0.0, -0.1, 0.3]
    mean = sum(x) / len(x)
    centred = [round(v - mean, 10) for v in x]
    assert round(mean, 10) == 0.1 and centred == [0.1, -0.1, -0.2, 0.2]
    assert abs(sum(centred)) < 1e-12
    counted = sum([0.1, 0.2, 0.2, -0.1]) / 4
    not_counted = sum([-0.2, -0.1, 0.1, -0.1]) / 4
    assert abs(counted) >= abs(mean) - 1e-12 and abs(not_counted) < abs(mean)
    assert round(not_counted, 10) == -0.075
    # p = count / R: RESULTS.md (step 11) prints p = 0.0027 for the improvement on SciFact
    assert "+0.0433  p = 0.0027" in (REPO / "RESULTS.md").read_text(encoding="utf-8")
    _assert_in_paper("27 of the 10,000", "$p = 27 / 10{,}000 = 0.0027$")


def test_auc_and_threshold_example():
    hits, misses = [0.9, 0.8, 0.6], [0.7, 0.2]
    rows = [{"gold_in_passages": True, "z": v} for v in hits] + \
           [{"gold_in_passages": False, "z": v} for v in misses]
    a = auc(hits, misses)
    assert a == 5 / 6
    t = best_threshold(rows, "z")
    out = apply_threshold(rows, "z", t)
    assert t == 0.8 and out["misses_caught"] == 2 and out["hits_refused"] == 1
    j = out["misses_caught"] / out["misses"] - out["hits_refused"] / out["hits"]
    _assert_in_paper(f"= 5/6 = {_shown(a, 2)}$", "$\\theta = 0.8$", f"= {_shown(j, 2)}$")


def test_real_threshold_example():
    test = json.loads((REPO / "results" / "abstention.json").read_text(encoding="utf-8"))
    out = test["signals"]["cross_encoder_top"]["threshold_from_dev_on_test"]
    caught, misses, refused, hits = out["misses_caught"], out["misses"], out["hits_refused"], out["hits"]
    j = caught / misses - refused / hits
    _assert_in_paper(f"$J = {caught}/{misses} - {refused}/{hits} = {_shown(caught / misses, 3)} - "
                     f"{_shown(refused / hits, 3)} = {_shown(j, 3)}$")
