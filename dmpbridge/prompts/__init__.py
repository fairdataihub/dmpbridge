"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
labels      LABELS tuple and OUTPUT_SCHEMA (Ollama structured-output enforcement)
system      SYSTEM_PROMPT shared by all strategies
"""
from .labels import LABELS, OUTPUT_SCHEMA
from .system import SYSTEM_PROMPT, build_system_prompt
from .few_shot import build_few_shot_examples

__all__ = ["LABELS", "OUTPUT_SCHEMA", "SYSTEM_PROMPT", "build_system_prompt", "build_few_shot_examples"]
