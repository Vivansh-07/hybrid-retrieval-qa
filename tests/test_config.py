"""Unit tests for config attribute access, --set overrides, output folders and the embedding cache key."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.config import Config, apply_overrides, load_config, output_dir  # noqa: E402
from hybridqa.pipeline import RetrievalPipeline  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    # point every folder a test might create at pytest's temp dir, not the repo
    base = load_config()
    base = base.override("paths.index_dir", str(tmp_path / "indexes"))
    return base.override("paths.results_dir", str(tmp_path / "results"))


def test_override_values_are_parsed_as_yaml(cfg):
    new = apply_overrides(cfg, ["fusion.k=30", "dense.normalize=false",
                                "fusion.weights=[1.0, 0.5]", "chunking.strategy=none"])
    assert new.get_path("fusion.k") == 30
    assert new.get_path("dense.normalize") is False
    assert new.get_path("fusion.weights") == [1.0, 0.5]
    assert new.get_path("chunking.strategy") == "none"  # a string, not YAML null


def test_override_does_not_modify_the_original(cfg):
    apply_overrides(cfg, ["fusion.k=30"])
    assert cfg.get_path("fusion.k") == 60


def test_override_rejects_unknown_keys(cfg):
    # a typo must fail loudly, not silently run the default configuration
    with pytest.raises(KeyError, match="stratgy"):
        apply_overrides(cfg, ["chunking.stratgy=none"])


def test_override_rejects_missing_equals(cfg):
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["chunking.strategy"])


def test_output_dir_keeps_side_experiments_out_of_the_main_results(cfg, tmp_path):
    main = output_dir(cfg)
    side = output_dir(cfg, overrides=["chunking.strategy=none"])
    assert main == tmp_path / "results"
    assert side == main / "set_chunking.strategy=none"
    assert output_dir(cfg, out=str(tmp_path / "custom")) == tmp_path / "custom"


def test_embedding_cache_changes_with_chunking_and_model(cfg):
    default = RetrievalPipeline(cfg, verbose=False).embedding_cache_path()
    whole_docs = RetrievalPipeline(apply_overrides(cfg, ["chunking.strategy=none"]),
                                   verbose=False).embedding_cache_path()
    other_model = RetrievalPipeline(apply_overrides(cfg, ["dense.model=BAAI/bge-small-en-v1.5"]),
                                    verbose=False).embedding_cache_path()
    assert len({default, whole_docs, other_model}) == 3


def test_embedding_cache_ignores_settings_that_do_not_change_the_vectors(cfg):
    # top_k only affects search, not the stored passage embeddings
    a = RetrievalPipeline(cfg, verbose=False).embedding_cache_path()
    b = RetrievalPipeline(apply_overrides(cfg, ["dense.top_k=50"]), verbose=False).embedding_cache_path()
    assert a == b


def test_string_settings_are_taken_literally(cfg):
    # BGE's query prefix ends in ": " — as YAML that would parse as a mapping, not a string
    prefix = "Represent this sentence for searching relevant passages: "
    new = apply_overrides(cfg, ["dense.model=BAAI/bge-small-en-v1.5", f"dense.query_prefix={prefix}"])
    assert new.get_path("dense.model") == "BAAI/bge-small-en-v1.5"
    assert new.get_path("dense.query_prefix") == prefix  # trailing space kept


def test_local_model_folder_is_resolved_and_tied_to_its_weights(cfg, tmp_path):
    import os
    from hybridqa.config import resolve_model

    assert resolve_model("BAAI/bge-small-en-v1.5") == "BAAI/bge-small-en-v1.5"  # a Hub id is left alone

    model_dir = tmp_path / "my-finetuned-model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"v1")
    assert resolve_model(str(model_dir)) == str(model_dir)

    local_cfg = apply_overrides(cfg, [f"dense.model={model_dir}"])
    before = RetrievalPipeline(local_cfg, verbose=False).embedding_cache_path()
    os.utime(weights, ns=(1, 1))  # "retrain": the weights file changes
    after = RetrievalPipeline(local_cfg, verbose=False).embedding_cache_path()
    assert before != after  # stale embeddings from the old weights are not reused
    assert ":" not in before.name  # Windows drive letters are sanitised out of the file name


def test_keep_tail_false_keeps_the_existing_embedding_cache(cfg):
    # issue #45: adding the setting must not rename caches built before it existed
    without = Config({**cfg, "chunking": {k: v for k, v in cfg["chunking"].items() if k != "keep_tail"}})
    off = RetrievalPipeline(cfg, verbose=False).embedding_cache_path()
    on = RetrievalPipeline(apply_overrides(cfg, ["chunking.keep_tail=true"]), verbose=False).embedding_cache_path()
    assert RetrievalPipeline(without, verbose=False).embedding_cache_path() == off != on


@pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin"])
def test_embedding_cache_follows_retrained_weights(cfg, tmp_path, weights):
    # issue #47: retraining a local model into the same folder must not reuse old vectors
    import os

    model = tmp_path / "my-model"
    model.mkdir()
    (model / weights).write_bytes(b"weights")
    local = apply_overrides(cfg, [f"dense.model={model}"])
    before = RetrievalPipeline(local, verbose=False).embedding_cache_path()
    later = (model / weights).stat().st_mtime_ns + 10**9
    os.utime(model / weights, ns=(later, later))
    assert RetrievalPipeline(local, verbose=False).embedding_cache_path() != before


def _read_sections_as_attributes(cfg):
    # attribute access now swaps each section for a Config in place; check that it did,
    # so the tests below really run on a config that has been read this way
    for name in ("dataset", "chunking", "dense", "paths"):
        assert getattr(cfg, name) is cfg[name] and type(cfg[name]) is Config


def test_attribute_access_returns_the_same_section_every_time(cfg):
    # issue #50: a write made through cfg.dense was lost, because each access built a new copy
    assert cfg.dense is cfg.dense is cfg["dense"]
    cfg.dense["top_k"] = 50
    assert cfg.get_path("dense.top_k") == 50


def test_attribute_assignment_is_refused_and_changes_nothing(cfg):
    # issue #50: cfg.dense.top_k = 50 used to set a Python attribute that get_path never saw
    top_k = cfg.get_path("dense.top_k")
    with pytest.raises(AttributeError, match=r"override\(\"dense\.top_k\", 50\).*--set dense\.top_k=50"):
        cfg.dense.top_k = 50
    with pytest.raises(AttributeError):
        cfg.new_section = {}
    assert cfg.get_path("dense.top_k") == top_k and cfg.dense.top_k == top_k
    assert not hasattr(cfg, "new_section") and "new_section" not in cfg


def test_missing_names_behave_like_missing_attributes(cfg):
    # issue #50: hasattr/getattr must still answer from the dict
    _read_sections_as_attributes(cfg)
    assert not hasattr(cfg, "no_such_section") and not hasattr(cfg.dense, "no_such_setting")
    assert getattr(cfg.dense, "no_such_setting", "default") == "default"


def test_deepcopy_and_pickle_after_attribute_access(cfg):
    # issue #50: override() deep-copies the config, so a swapped-in section must copy cleanly
    import copy
    import pickle

    _read_sections_as_attributes(cfg)
    for clone in (copy.deepcopy(cfg), pickle.loads(pickle.dumps(cfg))):
        assert type(clone) is Config and clone == cfg
        assert clone.dense is clone["dense"] and clone.dense is not cfg.dense
        clone.dense["top_k"] = 1
        assert clone.get_path("dense.top_k") == 1 != cfg.get_path("dense.top_k")


def test_saved_config_after_attribute_access_is_plain_json(cfg):
    # issue #50: save_results writes json.dumps(dict(cfg)); a write through cfg.dense must be in it
    _read_sections_as_attributes(cfg)
    cfg.dense["top_k"] = 50
    saved = json.loads(json.dumps(dict(cfg)))
    assert saved == cfg and saved["dense"]["top_k"] == 50


def test_override_after_attribute_access_leaves_the_original_unchanged(cfg):
    # issue #50: override() must still copy a section that was swapped for a Config
    _read_sections_as_attributes(cfg)
    top_k = cfg.dense.top_k
    new = cfg.override("dense.top_k", top_k + 1)
    assert new.dense.top_k == top_k + 1 and cfg.dense.top_k == top_k
    assert new.dense is not cfg.dense


def test_embedding_cache_is_the_same_after_attribute_access(cfg):
    # issue #50: swapping a section for a Config must not change the cache file name
    before = RetrievalPipeline(cfg, verbose=False).embedding_cache_path()
    _read_sections_as_attributes(cfg)
    assert RetrievalPipeline(cfg, verbose=False).embedding_cache_path() == before
