"""The demo's Results tab and its --best settings come from the committed files.

Running the demo needs the models, which GitHub's test machines do not have, but the
parts that read results/ and paper/figures/ do not. These tests catch a renamed results
folder or figure, or a changed fusion weight, before the presentation instead of during it.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo", REPO / "scripts" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_results_tab_shows_every_configuration(demo):
    table = demo.configs_table(REPO / "results").set_index("configuration")
    assert list(table.index) == [label for label, _ in demo.CONFIGS], "a results folder is missing"
    best = table.loc[demo.CONFIGS[-1][0], demo.LABELS["fusion"]]
    assert f"{best:.3f}" == "0.759"


def test_results_tab_figures_exist(demo):
    missing = [f for f in demo.FIGURES if not (REPO / "paper" / "figures" / f).exists()]
    assert not missing, f"the Results tab would leave out {missing}"


def test_best_uses_the_weight_chosen_on_dev(demo, tmp_path, monkeypatch):
    (tmp_path / demo.BEST_MODEL).mkdir(parents=True)  # stands in for the fine-tuned model
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    assert demo.best_overrides() == [f"dense.model={demo.BEST_MODEL}", "fusion.weights=[0.3, 1.0]"]


def test_best_without_the_model_says_what_to_run(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="finetune_dense.py"):
        demo.best_overrides()


def test_prepared_claims_are_test_claims(demo):
    test_ids = json.loads((REPO / "results" / "query_subsets.json").read_text(encoding="utf-8"))["properties"]
    ids = [re.search(r"\((\d+)\)$", label).group(1) for label, _ in demo.PREPARED]
    assert set(ids) <= set(test_ids), "every claim button should name a SciFact test claim id"
