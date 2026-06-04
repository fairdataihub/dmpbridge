"""
llm_narrative_blocks.py

Purpose:
Convert existing PDFPlumber-extracted DMP blocks into structured narrative blocks
using an LLM-primary, text-only, chunked whole-document labeling approach.

This version is for testing how well Llama can detect narrative structure
WITHOUT visual/layout metadata such as font size, bold, x/y position, color, etc.

Main idea:
- Use PDFPlumber text blocks.
- Preserve only text order and simple hierarchy/context hints.
- Process blocks in overlapping chunks.
- Ask the LLM to label each block as:
  - document_title
  - section
  - subsection
  - content
- Python only cleans, validates, merges, and records diagnostics.
"""

import json
import re
from pathlib import Path

from json_repair import repair_json

from dmpbridge.processing.text_cleaner import (
    clean_blocks,
    clean_repeated_words,
)


ALLOWED_LABELS = {
    "document_title",
    "section",
    "subsection",
    "content",
}


# Basic normalization helpers
# ------------------------------------------------------------

def normalize_text_simple(text: str) -> str:
    """Normalize spaces but keep the original wording."""
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_label(label: str) -> str:
    """Keep only allowed labels."""
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def is_page_number(text: str) -> bool:
    """Remove standalone page numbers."""
    return bool(re.match(r"^\d+$", text.strip()))


def is_punctuation_only(text: str) -> bool:
    """Remove blocks that contain only punctuation."""
    return bool(re.match(r"^[^\w]+$", text.strip()))


# Text-only hierarchy/context helpers
# ------------------------------------------------------------

def get_text_position_hint(index: int, total_blocks: int) -> str:
    """
    Simple document-order hint.

    This is not visual/layout information.
    It only tells the LLM whether the block appears near the beginning,
    middle, or end of the text sequence.
    """
    if total_blocks <= 0:
        return "unknown"

    ratio = index / max(total_blocks - 1, 1)

    if ratio <= 0.05:
        return "beginning"
    if ratio <= 0.25:
        return "early"
    if ratio <= 0.75:
        return "middle"

    return "late"


def get_text_shape_hint(text: str) -> dict:
    """
    Text-only clues that may help the LLM understand hierarchy.

    These are based only on the text itself.
    They are not visual features.
    """
    text = normalize_text_simple(text)
    words = text.split()

    starts_with_number = bool(
        re.match(r"^\s*(\d+[\.\)]|\d+\.\d+[\.\)]?)\s+", text)
    )

    looks_like_question = text.endswith("?")

    short_text = len(words) <= 12
    medium_text = 13 <= len(words) <= 35
    long_text = len(words) > 35

    has_colon = ":" in text
    ends_with_colon = text.endswith(":")

    # Examples:
    # 1. Data sharing and preservation
    # 2. Data used in publications
    numbered_heading_like = bool(
        re.match(r"^\s*\d+[\.\)]\s+[A-Z]", text)
    )

    # Examples:
    # Data Types and Sources.
    # Roles & Responsibilities.
    title_case_like = (
        len(words) <= 12
        and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) // 2)
    )

    return {
        "word_count": len(words),
        "character_count": len(text),
        "starts_with_number": starts_with_number,
        "numbered_heading_like": numbered_heading_like,
        "looks_like_question": looks_like_question,
        "short_text": short_text,
        "medium_text": medium_text,
        "long_text": long_text,
        "has_colon": has_colon,
        "ends_with_colon": ends_with_colon,
        "title_case_like": title_case_like,
    }


def compact_blocks_for_labeling(pdf_blocks: list[dict]) -> list[dict]:
    """
    Convert PDFPlumber blocks into compact text-only blocks for LLM labeling.

    Removed from this version:
    - font_size
    - body_font_size
    - relative_font
    - is_bold
    - visual_hint
    - x0/x1/top/bottom
    - indent
    - vertical_gap_before/after

    Kept:
    - block_id
    - text
    - page if available
    - line_order if available
    - text_position_hint
    - text_shape_hint
    """
    pdf_blocks = clean_blocks(pdf_blocks)

    clean_pdf_blocks = []

    for block in pdf_blocks:
        if not isinstance(block, dict):
            continue

        text = normalize_text_simple(block.get("text", ""))

        if not text:
            continue

        if is_page_number(text) or is_punctuation_only(text):
            continue

        clean_block = dict(block)
        clean_block["text"] = clean_repeated_words(text)
        clean_pdf_blocks.append(clean_block)

    compact_blocks = []
    total_blocks = len(clean_pdf_blocks)

    for i, block in enumerate(clean_pdf_blocks):
        text = normalize_text_simple(block.get("text", ""))

        compact_blocks.append(
            {
                "block_id": i + 1,
                "text": text,
                "page": block.get("page"),
                "line_order": block.get("line_order"),
                "text_position_hint": get_text_position_hint(
                    index=i,
                    total_blocks=total_blocks,
                ),
                "text_shape_hint": get_text_shape_hint(text),
            }
        )

    return compact_blocks


# Chunking
# ------------------------------------------------------------

def make_overlapping_chunks(
    blocks: list[dict],
    chunk_size: int = 45,
    overlap: int = 5,
) -> list[dict]:
    """
    Split blocks into overlapping chunks.

    The target area is labeled.
    The overlap gives the LLM nearby context.
    """
    chunks = []

    if not blocks:
        return chunks

    start = 0
    chunk_id = 1

    while start < len(blocks):
        end = min(start + chunk_size, len(blocks))

        context_start = max(0, start - overlap)
        context_end = min(len(blocks), end + overlap)

        chunk_blocks = blocks[context_start:context_end]
        target_ids = [
            block["block_id"]
            for block in blocks[start:end]
        ]

        chunks.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "context_start": context_start,
                "context_end": context_end,
                "target_block_ids": target_ids,
                "blocks": chunk_blocks,
            }
        )

        if end >= len(blocks):
            break

        start = end
        chunk_id += 1

    return chunks


# Prompt
# ------------------------------------------------------------

def build_chunk_labeling_prompt(
    chunk: dict,
    total_blocks: int,
) -> str:
    """
    Build the LLM prompt for one chunk.

    This prompt intentionally uses text-only information.
    It does not mention font, bold, color, spacing, indentation, or layout.
    """
    blocks_text = json.dumps(
        chunk["blocks"],
        indent=2,
        ensure_ascii=False,
    )

    target_ids_text = json.dumps(
        chunk["target_block_ids"],
        ensure_ascii=False,
    )

    return f"""
You are an expert at reading Data Management Plans (DMPs).

Your task:
You are given one chunk of a larger DMP document.

The chunk includes:
1. context blocks before and after the target area
2. target blocks that must be labeled

Read the full chunk to understand the local document hierarchy.
Use the context blocks only for understanding.
Return labels ONLY for the target_block_ids.

Document information:
- total_blocks_in_document: {total_blocks}
- chunk_id: {chunk["chunk_id"]}
- target_block_ids: {target_ids_text}

Each block includes text-only information:
- block_id
- text
- page, if available
- line_order, if available
- text_position_hint
- text_shape_hint

Do NOT rely on visual or layout features.
This test is designed to evaluate how well the LLM detects structure from text and hierarchy only.

Allowed labels:
- document_title
- section
- subsection
- content

Label meanings:

document_title:
The main title of the whole DMP or proposal-related DMP document.
Usually appears near the beginning of the document.
Only use this for the overall document title, not regular headings.

section:
A major heading that opens a new DMP topic, requirement area, or major document part.
Examples may include numbered headings such as:
"1. Data sharing and preservation"
"2. Data used in publications"
"3. Data management resources"

subsection:
A smaller prompt, question, or internal heading that belongs under a major section.
Examples may include short text prompts such as:
"Data Types and Sources."
"Content and Format."
"Roles & Responsibilities."
"Data Repositories."
"Data Volume."

content:
Narrative body text, answers, explanations, guidance, examples, list items,
data descriptions, institution names, repository descriptions, continuation lines,
and anything that is not clearly a title, section, or subsection.

Important instructions:
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT explain your reasoning.
- Do NOT rewrite text.
- Do NOT summarize text.
- Do NOT invent headings.
- Return labels ONLY for target_block_ids.
- Do NOT return labels for context-only blocks.
- Do NOT omit any target_block_id.
- Do NOT add new block_ids.
- Do NOT change block order.
- Use text and document hierarchy only.
- If uncertain, label the block as content.
- Prefer content over over-detecting headings.

Return format:
[
  {{"block_id": 1, "label": "document_title"}},
  {{"block_id": 2, "label": "section"}},
  {{"block_id": 3, "label": "content"}}
]

Chunk blocks:
{blocks_text}
"""

# LLM output parsing
# ------------------------------------------------------------

def extract_json_array(model_output: str) -> str:
    """Extract JSON array from the model output."""
    if not model_output or not str(model_output).strip():
        raise ValueError("LLM returned empty output.")

    model_output = str(model_output).strip()

    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        print("\nLLM raw output preview:")
        print(model_output[:2000])
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def parse_llm_labels(model_text: str) -> list[dict]:
    """Parse LLM labels and repair JSON if needed."""
    json_text = extract_json_array(model_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        parsed = json.loads(repaired)

    if not isinstance(parsed, list):
        raise ValueError("LLM output JSON is not a list.")

    return parsed


def labels_to_lookup(
    llm_labels: list[dict],
    block_lookup: dict[int, dict],
    allowed_block_ids: set[int] | None = None,
) -> dict[int, str]:
    """Convert LLM output into block_id -> label dictionary."""
    label_lookup = {}

    for item in llm_labels:
        if not isinstance(item, dict):
            continue

        block_id = item.get("block_id")
        label = normalize_label(item.get("label", ""))

        try:
            block_id = int(block_id)
        except (TypeError, ValueError):
            continue

        if allowed_block_ids is not None and block_id not in allowed_block_ids:
            continue

        if block_id in block_lookup and label:
            label_lookup[block_id] = label

    return label_lookup



# Diagnostics
# ------------------------------------------------------------

def build_labeling_diagnostics(
    compact_blocks: list[dict],
    label_lookup: dict[int, str],
    chunk_diagnostics: list[dict] | None = None,
) -> dict:
    """Create diagnostics about missing labels and label counts."""
    expected_ids = {block["block_id"] for block in compact_blocks}
    returned_ids = set(label_lookup.keys())
    missing_ids = sorted(expected_ids - returned_ids)

    label_counts = {label: 0 for label in ALLOWED_LABELS}

    for label in label_lookup.values():
        if label in label_counts:
            label_counts[label] += 1

    return {
        "total_blocks": len(compact_blocks),
        "labeled_blocks": len(returned_ids),
        "missing_label_count": len(missing_ids),
        "missing_block_ids": missing_ids,
        "label_counts": label_counts,
        "chunk_diagnostics": chunk_diagnostics or [],
    }



# Post-processing
# ------------------------------------------------------------

def postprocess_blocks(blocks: list[dict]) -> list[dict]:
    """
    Clean final blocks after LLM labeling.

    This version does not add rule-based heading correction.
    That keeps the experiment focused on Llama's text-only labeling ability.
    """
    clean_output = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        label = normalize_label(block.get("label", ""))
        text = normalize_text_simple(block.get("text", ""))

        if not label or not text:
            continue

        text = clean_repeated_words(text)

        if is_page_number(text):
            continue

        if is_punctuation_only(text):
            continue

        clean_output.append(
            {
                "label": label,
                "text": text,
            }
        )

    return clean_output


def merge_consecutive_document_title_blocks(blocks: list[dict]) -> list[dict]:
    """Merge consecutive title blocks if LLM splits the title."""
    merged = []

    for block in blocks:
        if (
            merged
            and block["label"] == "document_title"
            and merged[-1]["label"] == "document_title"
        ):
            merged[-1]["text"] += " " + block["text"]
        else:
            merged.append(dict(block))

    return merged


def merge_consecutive_content_blocks(blocks: list[dict]) -> list[dict]:
    """
    Merge consecutive content blocks.

    This makes the final JSON easier to read.
    It does not change title/section/subsection labels.
    """
    merged = []

    for block in blocks:
        if (
            merged
            and block["label"] == "content"
            and merged[-1]["label"] == "content"
        ):
            previous_text = merged[-1]["text"]
            current_text = block["text"]

            if previous_text.endswith("-"):
                merged[-1]["text"] = previous_text[:-1] + current_text

            elif previous_text.endswith((".", ":", ";", "?", "!")):
                merged[-1]["text"] += "\n\n" + current_text

            else:
                merged[-1]["text"] += " " + current_text

        else:
            merged.append(dict(block))

    return merged



# Main LLM labeling functions
# ------------------------------------------------------------

def label_blocks_with_chunked_llm(
    llm,
    compact_blocks: list[dict],
    chunk_size: int = 45,
    overlap: int = 5,
) -> tuple[dict[int, str], list[dict]]:
    """Run LLM labeling on all chunks."""
    block_lookup = {
        block["block_id"]: block
        for block in compact_blocks
    }

    chunks = make_overlapping_chunks(
        blocks=compact_blocks,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    final_label_lookup = {}
    chunk_diagnostics = []

    for chunk in chunks:
        target_ids = set(chunk["target_block_ids"])

        prompt = build_chunk_labeling_prompt(
            chunk=chunk,
            total_blocks=len(compact_blocks),
        )

        try:
            response = llm.invoke(prompt)
            model_text = response.content if hasattr(response, "content") else str(response)

            if not model_text or not model_text.strip():
                raise ValueError("LLM returned empty response.")

            llm_labels = parse_llm_labels(model_text)

            chunk_label_lookup = labels_to_lookup(
                llm_labels=llm_labels,
                block_lookup=block_lookup,
                allowed_block_ids=target_ids,
            )

            final_label_lookup.update(chunk_label_lookup)

            missing_targets = sorted(target_ids - set(chunk_label_lookup.keys()))

            chunk_diagnostics.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "target_count": len(target_ids),
                    "labeled_count": len(chunk_label_lookup),
                    "missing_count": len(missing_targets),
                    "missing_block_ids": missing_targets,
                    "status": "ok",
                }
            )

        except Exception as e:
            chunk_diagnostics.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "target_count": len(target_ids),
                    "labeled_count": 0,
                    "missing_count": len(target_ids),
                    "missing_block_ids": sorted(target_ids),
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )

    return final_label_lookup, chunk_diagnostics


def generate_structured_blocks_with_llm_labels_only(
    llm,
    pdf_blocks: list[dict],
    return_diagnostics: bool = False,
    chunk_size: int = 45,
    overlap: int = 5,
):
    """
    Generate final structured narrative blocks using LLM labels only.

    This version intentionally avoids visual/layout metadata.
    """
    compact_blocks = compact_blocks_for_labeling(pdf_blocks)

    block_lookup = {
        block["block_id"]: block
        for block in compact_blocks
    }

    if not compact_blocks:
        empty_result = []
        diagnostics = {
            "total_blocks": 0,
            "labeled_blocks": 0,
            "missing_label_count": 0,
            "missing_block_ids": [],
            "label_counts": {label: 0 for label in ALLOWED_LABELS},
            "chunk_diagnostics": [],
        }

        if return_diagnostics:
            return empty_result, diagnostics

        return empty_result

    label_lookup, chunk_diagnostics = label_blocks_with_chunked_llm(
        llm=llm,
        compact_blocks=compact_blocks,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    diagnostics = build_labeling_diagnostics(
        compact_blocks=compact_blocks,
        label_lookup=label_lookup,
        chunk_diagnostics=chunk_diagnostics,
    )

    rebuilt_blocks = []

    for block_id in sorted(block_lookup):
        original_block = block_lookup[block_id]
        text = normalize_text_simple(original_block.get("text", ""))

        if not text:
            continue

        # If LLM missed a label, default to content.
        label = label_lookup.get(block_id, "content")

        rebuilt_blocks.append(
            {
                "label": label,
                "text": clean_repeated_words(text),
            }
        )

    blocks = postprocess_blocks(rebuilt_blocks)
    blocks = merge_consecutive_document_title_blocks(blocks)
    blocks = merge_consecutive_content_blocks(blocks)

    if return_diagnostics:
        return blocks, diagnostics

    return blocks


def generate_structured_blocks_with_llm(
    llm,
    pdf_blocks: list[dict],
    return_diagnostics: bool = False,
    chunk_size: int = 45,
    overlap: int = 5,
):
    """Compatibility wrapper."""
    return generate_structured_blocks_with_llm_labels_only(
        llm=llm,
        pdf_blocks=pdf_blocks,
        return_diagnostics=return_diagnostics,
        chunk_size=chunk_size,
        overlap=overlap,
    )

# Saving
# ------------------------------------------------------------

def save_blocks(blocks: list[dict], output_path):
    """Save final structured blocks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)


def save_diagnostics(diagnostics: dict, output_path):
    """Save diagnostics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
