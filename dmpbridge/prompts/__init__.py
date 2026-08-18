"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
labels              LABELS tuple and OUTPUT_SCHEMA (Ollama structured-output enforcement)
system              SYSTEM_PROMPT shared by docling/lighton's id-based per-block payload
pdfplumber_visual   VISUAL_SIGNAL_SYSTEM_PROMPT / VISUAL_SIGNAL_SCHEMA — pdfplumber's
                    whole-document, marker-based, id-less path (see WholeDocStrategy)
"""
from .labels import LABELS, OUTPUT_SCHEMA
from .pdfplumber_visual import VISUAL_SIGNAL_SCHEMA, VISUAL_SIGNAL_SYSTEM_PROMPT
from .system import SYSTEM_PROMPT

__all__ = [
    "LABELS", "OUTPUT_SCHEMA", "SYSTEM_PROMPT",
    "VISUAL_SIGNAL_SYSTEM_PROMPT", "VISUAL_SIGNAL_SCHEMA",
]
