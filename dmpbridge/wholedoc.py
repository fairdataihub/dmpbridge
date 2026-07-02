"""Shared helpers for whole-document inference.

These utilities are used by :class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy`
to convert extracted blocks into the API payload format, parse model responses,
and merge predicted labels back into the block list.
"""
import json
from pathlib import Path

from .logging_setup import get_logger
from .prompt import LABELS

logger = get_logger(__name__)


def build_payload(blocks: list[dict]) -> list[dict]:
    """Convert extracted blocks into the model payload format.

    Adds a numeric ``id`` field and surfaces ``bold`` / ``italic`` flags so the
    model can use visual cues when assigning labels.
    """
    return [
        {
            "id":     j,
            "text":   b["text"],
            "bold":   b["is_bold"],
            "italic": b.get("is_italic", False),
            "page":   b["page"],
        }
        for j, b in enumerate(blocks)
    ]


def parse_response(raw: str, label: str = "") -> list[dict]:
    """Parse the model's JSON response into a list of ``{id, label}`` dicts.

    Handles markdown code fences and dict-wrapped arrays gracefully.
    Returns an empty list on parse failure.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts   = cleaned.split("```", 2)
        cleaned = parts[1].lstrip("json").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("%s JSON parse error: %s", label, e)
        return []


def apply_labels(blocks: list[dict], parsed: list[dict]) -> list[dict]:
    """Merge model predictions back into the extracted block list.

    Predictions are matched by the numeric ``id`` set in :func:`build_payload`.
    Blocks with no prediction or an invalid label default to ``answer.text``.
    """
    result = [dict(b) for b in blocks]
    for entry in parsed:
        idx = entry.get("id")
        lbl = entry.get("label", "answer.text")
        if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
            result[idx]["label"] = lbl
    for b in result:
        if not b.get("label"):
            b["label"] = "answer.text"
    return result
