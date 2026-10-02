"""PDF -> RDA maDMP JSON: prompt versions, scoring against a hand-made reference,
and comparison across versions. Used by notebooks/rda-dmp-json.ipynb."""
from dmpbridge.rda.schema import load_schema
from dmpbridge.rda.score import score, metrics, reference_fields
from dmpbridge.rda.version import Version

__all__ = ["load_schema", "score", "metrics", "reference_fields", "Version"]
