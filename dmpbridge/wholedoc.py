"""Shared helpers for whole-document inference scripts."""
import json
from pathlib import Path
from typing import Callable

from .classifier import LABELS
from .extractor import extract_blocks
from .logging_setup import get_logger

logger = get_logger(__name__)


def build_payload(blocks: list[dict]) -> list[dict]:
    """Convert extracted blocks into the API payload format."""
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


def build_prompt(payload: list[dict]) -> str:
    """Format the single-call classification prompt for a full document."""
    return (
        f"CLASSIFY ALL BLOCKS — return a JSON array with exactly {len(payload)} entries, "
        f"one per block, in the same order:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def parse_response(raw: str, label: str = "") -> list[dict]:
    """Parse the model's JSON response.

    Handles markdown code fences (Claude sometimes wraps output in ```json blocks)
    and dict-wrapped arrays (model returns {"blocks": [...]} instead of [...]).
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

    Predictions are matched by the numeric id field assigned in build_payload.
    Any block with no prediction or an invalid label defaults to 'answer.text'.
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


def run_samples(
    pdf_dir: Path,
    out_dir: Path,
    classify_fn: Callable,
    tag: str,
    sample_range: range = range(1, 11),
) -> None:
    """Run whole-document inference for each sample index in sample_range.

    Parameters
    ----------
    classify_fn : callable
        Provider-specific function with signature:
        ``classify_fn(blocks, payload, prompt, label) -> list[dict]``
        Called once per sample; returns the list of parsed prediction entries.
    tag : str
        Output filename suffix: ``out_dir/sample{N}_{tag}.json``
    sample_range : range
        Indices to process (default ``range(1, 11)`` = samples 1–10).
    """
    for i in sample_range:
        label    = f"[sample{i}]"
        pdf_path = pdf_dir / f"sample{i}.pdf"
        out_path = out_dir / f"sample{i}_{tag}.json"

        if out_path.exists():
            logger.info("%s already exists — skipping", label)
            continue

        logger.info("%s extracting blocks from %s …", label, pdf_path.name)
        blocks  = extract_blocks(pdf_path)
        payload = build_payload(blocks)
        prompt  = build_prompt(payload)

        parsed = classify_fn(blocks, payload, prompt, label)
        result = apply_labels(blocks, parsed)

        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("%s saved → %s", label, out_path.name)

    logger.info("Done.")
