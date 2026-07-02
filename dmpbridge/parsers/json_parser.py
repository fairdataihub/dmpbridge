"""Unified parser for LLM JSON responses.

LLM output is messy: markdown code fences, dict-wrapped arrays, trailing
commas, truncated JSON.  This module centralises all that handling so each
strategy and classifier imports one function instead of duplicating the logic.
"""
import json

from ..utils import get_logger

logger = get_logger(__name__)


def parse_llm_json(raw: str, label: str = "") -> list[dict]:
    """Parse a raw LLM response into a list of dicts.

    Handles the three most common failure modes:

    1. **Markdown fences** — the model wraps its output in ```json … ```.
    2. **Dict-wrapped arrays** — the model returns ``{"labels": [...]}``
       instead of a bare ``[...]``.
    3. **Empty response** — returns an empty list with a warning.

    Parameters
    ----------
    raw:
        The raw text returned by the model.
    label:
        An optional context string (e.g. sample name or batch offset) included
        in warning log messages to make debugging easier.

    Returns
    -------
    list[dict]
        Parsed list of entry dicts, or ``[]`` on any parse failure.
    """
    if not raw or not raw.strip():
        logger.warning("%s empty response from model", label)
        return []

    cleaned = raw.strip()

    # Strip markdown code fences (``` or ```json).
    if cleaned.startswith("```"):
        parts   = cleaned.split("```", 2)
        cleaned = parts[1].lstrip("json").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("%s JSON parse error: %s", label, exc)
        return []

    # Unwrap {"labels": [...]} or any single-key dict wrapping an array.
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])

    if not isinstance(data, list):
        logger.warning("%s unexpected response shape (got %s)", label, type(data).__name__)
        return []

    return data
