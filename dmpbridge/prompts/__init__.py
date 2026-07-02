"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
labels      LABELS tuple and OUTPUT_SCHEMA (Ollama structured-output enforcement)
system      SYSTEM_PROMPT shared by all strategies and providers

Strategy-specific prompt builders live in their own strategy files:
    strategies/batch.py       _build_batch_prompt()
    strategies/wholedoc.py    _build_wholedoc_prompt()
    strategies/pdf_direct.py  _PDF_USER_PROMPT
"""
from .labels import LABELS, OUTPUT_SCHEMA
from .system import SYSTEM_PROMPT

__all__ = ["LABELS", "OUTPUT_SCHEMA", "SYSTEM_PROMPT"]
