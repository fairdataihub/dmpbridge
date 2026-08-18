"""Output directory layout — one folder per pipeline stage.

The pipeline runs in four stages, and each writes its own directory so any
stage can be inspected, diffed or re-run without touching the others::

    data/output/
    ├── 1_extracted/<extractor>/sampleN.json     text blocks, no labels
    ├── 2_labeled/<tag>/sampleN.json             the same blocks, labeled by the LLM
    ├── 3_structured/<tag>/sampleN.json          nested into the DMP Tool schema
    └── 4_final/<tag>/sampleN.json               Rules.xlsx conversion applied

``<tag>`` is ``<model-slug>_<extractor>_whole_doc``; ``<extractor>`` alone keys
stage 1, because extraction does not depend on the model.  That is the point of
splitting it out: extracting a document once and labeling it with three models
costs one extraction rather than three.

Stages 3 and 4 previously shared a directory with stage 2, distinguished only by
a ``_structured`` filename suffix.  Filenames are now identical across stages so
``sampleN.json`` can be compared directly from one folder to the next.
"""
from pathlib import Path

# Repository root — three levels up from this file (dmpbridge/core/paths.py).
_ROOT = Path(__file__).parent.parent.parent

OUTPUT_ROOT    = _ROOT / "data" / "output"

EXTRACTED_DIR  = OUTPUT_ROOT / "1_extracted"    # keyed by extractor
LABELED_DIR    = OUTPUT_ROOT / "2_labeled"      # keyed by tag
STRUCTURED_DIR = OUTPUT_ROOT / "3_structured"   # keyed by tag
FINAL_DIR      = OUTPUT_ROOT / "4_final"        # keyed by tag

STAGE_DIRS = {
    "extracted":  EXTRACTED_DIR,
    "labeled":    LABELED_DIR,
    "structured": STRUCTURED_DIR,
    "final":      FINAL_DIR,
}


def make_tag(model: str, extractor: str, strategy: str = "whole_doc") -> str:
    """Build the directory name identifying one run configuration.

    ``model`` may contain a colon (Ollama tags look like ``llama3.1:8b``), which
    is not valid in a Windows path, so it is replaced with a hyphen.
    """
    return f"{model.replace(':', '-')}_{extractor}_{strategy}"


def extracted_path(extractor: str, sample: int) -> Path:
    """Stage 1 — text blocks for one sample, shared by every model."""
    return EXTRACTED_DIR / extractor / f"sample{sample}.json"


def labeled_path(tag: str, sample: int) -> Path:
    """Stage 2 — LLM-labeled blocks."""
    return LABELED_DIR / tag / f"sample{sample}.json"


def structured_path(tag: str, sample: int) -> Path:
    """Stage 3 — nested DMP Tool schema."""
    return STRUCTURED_DIR / tag / f"sample{sample}.json"


def final_path(tag: str, sample: int) -> Path:
    """Stage 4 — Rules.xlsx conversion applied."""
    return FINAL_DIR / tag / f"sample{sample}.json"


def list_tags(stage: str = "structured") -> list[str]:
    """Return the run tags present in *stage*, sorted."""
    root = STAGE_DIRS[stage]
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and any(d.glob("sample*.json")))


__all__ = [
    "OUTPUT_ROOT", "EXTRACTED_DIR", "LABELED_DIR", "STRUCTURED_DIR", "FINAL_DIR",
    "STAGE_DIRS", "make_tag", "extracted_path", "labeled_path", "structured_path",
    "final_path", "list_tags",
]
