"""The worked examples in the full paper's Methodology (paper_full/sections/03_methodology.tex).

Each example is recomputed here, with the project's own code where it has a function for it,
and the paper must show the result as the test rounds it. A slip in the arithmetic, or a
rewrite that changes a number, fails here instead of in front of the examiner.
"""

import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
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


def test_passage_count_example():
    # m(n) = ceil((n - W) / S) + 1 windows for n > W sentences, W = 3, S = 2
    n, window, stride = 7, 3, 2
    starts = list(range(1, n - window + 2, stride))
    if starts[-1] + window - 1 < n:
        starts.append(starts[-1] + stride)
    assert len(starts) == math.ceil((n - window) / stride) + 1 == 3
    assert [(s, min(s + window - 1, n)) for s in starts] == [(1, 3), (3, 5), (5, 7)]
    _assert_in_paper("$\\lceil (7-3)/2 \\rceil + 1 = 3$", "sentences 1--3, 3--5 and 5--7")


def test_reordering_bound_example():
    # a BM25-first passage at dense rank r ties with the dense top passage when 1/(60+r) = 0.7/61
    r = 61 / 0.7 - 60
    assert round(r) == 27 and 1 / (60 + 27) + 0.3 / 61 > 1 / 61 > 1 / (60 + 28) + 0.3 / 61
    _assert_in_paper("r = \\frac{61}{0.7} - 60 \\approx 27", "up to about 26 places")


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
