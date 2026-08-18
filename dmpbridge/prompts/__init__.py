"""Shared prompt definitions used across all strategies and providers.

Sub-modules
-----------
constants   LABELS tuple, and pdfplumber's own SYSTEM_PROMPT / schema
            (whole-document, id-less) — access via
            ``from dmpbridge.prompts import constants`` then
            ``constants.SYSTEM_PROMPT`` / ``constants.schema``
"""
from . import constants
from .constants import LABELS

__all__ = ["LABELS", "constants"]
