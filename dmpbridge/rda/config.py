"""Read a pipeline YAML config with a ``narrative:`` block and an ``rda:`` block.

    narrative:
      model: gemma4:e4b       # stages 2-4
    rda:
      model: llama3.1:8b      # stages 5-6; leave the block out to skip them

The narrative block becomes the experiment runner's ``model`` (a plain top-level
``model:`` still works), and the rda block is returned separately, so the
experiment runner reads the same config it always has. Used by
scripts/run_demo.py and notebooks/demo-from-yaml-config.ipynb.
"""
from pathlib import Path
from typing import Optional

import yaml

from ..evaluation.experiment import ExperimentConfig


def _block_model(data: dict, key: str) -> Optional[str]:
    """The ``model`` of block *key*, or None when the block is absent."""
    block = data.pop(key, None)
    if block is None:
        return None
    if not (isinstance(block, dict) and block.get("model")):
        raise ValueError(f"{key}: must be a block with a model, "
                         f"e.g. '{key}: {{model: llama3.1:8b}}' - got {block!r}")
    return block["model"]


def load_pipeline_config(path) -> tuple[ExperimentConfig, Optional[str]]:
    """``(narrative config, rda model or None)`` from the YAML file at *path*."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    narrative_model = _block_model(data, "narrative")
    if narrative_model:
        if "model" in data or "models" in data:
            raise ValueError("give the narrative model once: in the narrative: block "
                             "or as model:, not both")
        data["model"] = narrative_model
    rda_model = _block_model(data, "rda")
    return ExperimentConfig.from_dict(data), rda_model
