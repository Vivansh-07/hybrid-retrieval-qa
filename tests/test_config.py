"""Unit tests for command-line config overrides, output folders and the embedding cache key."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridqa.config import apply_overrides, load_config, output_dir  # noqa: E402
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
