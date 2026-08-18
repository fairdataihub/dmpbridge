"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
labels              LABELS tuple and OUTPUT_SCHEMA (Ollama structured-output enforcement)
system              SYSTEM_PROMPT shared by docling/lighton's id-based per-block payload
pdfplumber_visual   SYSTEM_PROMPT / schema for pdfplumber's whole-document, id-less path
                    (see WholeDocStrategy.classify_entire_document) — not re-exported here
                    since its SYSTEM_PROMPT would collide with system.SYSTEM_PROMPT; import
                    the submodule directly: ``from dmpbridge.prompts import pdfplumber_visual``.
"""
from . import pdfplumber_visual
from .labels import LABELS, OUTPUT_SCHEMA
from .system import SYSTEM_PROMPT

__all__ = ["LABELS", "OUTPUT_SCHEMA", "SYSTEM_PROMPT", "pdfplumber_visual"]
