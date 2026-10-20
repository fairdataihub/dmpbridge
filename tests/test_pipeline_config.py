"""dmpbridge.rda.config: the narrative: and rda: blocks of a pipeline YAML config."""
import pytest

from dmpbridge.rda.config import load_pipeline_config

BASE = "name: x\nstrategy: wholedoc\nprovider: ollama\n"


def load(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(BASE + body, encoding="utf-8")
    return load_pipeline_config(path)


def test_both_blocks(tmp_path):
    cfg, rda_model = load(tmp_path, "narrative:\n  model: gemma4:e4b\nrda:\n  model: llama3.1:8b\n")
    assert cfg.models == ["gemma4:e4b"]
    assert rda_model == "llama3.1:8b"


def test_no_rda_block_means_narrative_only(tmp_path):
    cfg, rda_model = load(tmp_path, "narrative:\n  model: gemma4:e4b\n")
    assert cfg.models == ["gemma4:e4b"] and rda_model is None


def test_plain_model_still_works(tmp_path):
    cfg, rda_model = load(tmp_path, "model: gemma4:e4b\nrda:\n  model: llama3.1:8b\n")
    assert cfg.models == ["gemma4:e4b"] and rda_model == "llama3.1:8b"


def test_narrative_model_given_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="once"):
        load(tmp_path, "model: a\nnarrative:\n  model: b\n")


@pytest.mark.parametrize("body", ["rda: llama3.1:8b\n", "narrative: {}\n"])
def test_block_needs_a_model(tmp_path, body):
    with pytest.raises(ValueError, match="must be a block with a model"):
        load(tmp_path, body)
