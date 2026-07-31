"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
labels      LABELS tuple and OUTPUT_SCHEMA (Ollama structured-output enforcement)
system      SYSTEM_PROMPT shared by all strategies
"""
from .labels import LABELS, OUTPUT_SCHEMA
from .system import SYSTEM_PROMPT

__all__ = ["LABELS", "OUTPUT_SCHEMA", "SYSTEM_PROMPT"]
