"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
labels   LABELS tuple, OUTPUT_SCHEMA (lighton's id-based schema), and
         pdfplumber's own SYSTEM_PROMPT / schema (whole-document, id-less —
         kept here, not re-exported, since bare SYSTEM_PROMPT would collide
         with system.SYSTEM_PROMPT below; access via
         ``from dmpbridge.prompts import labels`` then
         ``labels.SYSTEM_PROMPT`` / ``labels.schema``)
system   SYSTEM_PROMPT shared by lighton's id-based per-block payload
"""
from . import labels
from .labels import LABELS, OUTPUT_SCHEMA
from .system import SYSTEM_PROMPT

__all__ = ["LABELS", "OUTPUT_SCHEMA", "SYSTEM_PROMPT", "labels"]
