"""
llm_narrative_blocks.py

Purpose:
Convert existing PDFPlumber-extracted DMP blocks into structured narrative blocks
using an LLM-primary, chunked whole-document labeling approach.

Best strategy for Llama 3.1 8B:
- Use existing PDFPlumber blocks.
- Preserve PDFPlumber text and metadata.
- Add layout metadata as context.
- Process blocks in overlapping chunks to avoid long prompts.
- Let the LLM assign labels.
- Python only cleans, validates, merges, and records diagnostics.

Final labels:
- document_title
- section
- subsection
- content
"""

import json
import re
from pathlib import Path
from statistics import median

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
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", text.strip()))


def is_punctuation_only(text: str) -> bool:
    return bool(re.match(r"^[^\w]+$", text.strip()))


def safe_float(value):
    try:
        if value is None:
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def get_layout_value(block: dict, key: str):
    return safe_float(block.get(key))


def get_font_size(block: dict) -> float:
    value = (
        block.get("avg_font_size")
        or block.get("font_size")
        or block.get("size")
        or 0
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_body_font_size(blocks: list[dict]) -> float:
    sizes = [
        get_font_size(block)
        for block in blocks
        if get_font_size(block) > 0 and str(block.get("text", "")).strip()
    ]

    return median(sizes) if sizes else 11.0


def is_bold_block(block: dict) -> bool:
    if block.get("is_bold") is True:
        return True

    font_names = block.get("font_names") or []
    font_name = str(block.get("font_name", "")).lower()

    all_fonts = " ".join(str(name).lower() for name in font_names)
    all_fonts = f"{all_fonts} {font_name}"

    return any(
        marker in all_fonts
        for marker in ["bold", "black", "heavy", "semibold", "demibold"]
    )


def get_relative_font_hint(font_size: float, body_font_size: float) -> str:
    if font_size <= 0 or body_font_size <= 0:
        return "unknown"

    ratio = font_size / body_font_size

    if ratio >= 1.20:
        return "much_larger_than_body"

    if ratio >= 1.08:
        return "slightly_larger_than_body"

    if ratio <= 0.90:
        return "smaller_than_body"

    return "similar_to_body"


def get_visual_hint(block: dict, body_font_size: float) -> str:
    text = normalize_text_simple(block.get("text", ""))
    font_size = get_font_size(block)
    relative_font = get_relative_font_hint(font_size, body_font_size)

    style_parts = []

    if is_bold_block(block):
        style_parts.append("bold")

    style_parts.append(relative_font)

    if text.isupper() and len(text.split()) <= 20:
        style_parts.append("uppercase_text")

    return "; ".join(style_parts)


def get_position_hint(index: int, total_blocks: int, page) -> str:
    if total_blocks <= 0:
        return "unknown"

    ratio = index / max(total_blocks - 1, 1)

    if ratio <= 0.05:
        position = "beginning"
    elif ratio <= 0.25:
        position = "early"
    elif ratio <= 0.75:
        position = "middle"
    else:
        position = "late"

    return f"{position}; page={page}"


def get_vertical_gap_before(blocks: list[dict], index: int):
    if index == 0:
        return None

    current_top = get_layout_value(blocks[index], "top")
    previous_bottom = get_layout_value(blocks[index - 1], "bottom")

    if current_top is None or previous_bottom is None:
        return None

    return round(current_top - previous_bottom, 2)


def get_vertical_gap_after(blocks: list[dict], index: int):
    if index + 1 >= len(blocks):
        return None

    current_bottom = get_layout_value(blocks[index], "bottom")
    next_top = get_layout_value(blocks[index + 1], "top")

    if current_bottom is None or next_top is None:
        return None

    return round(next_top - current_bottom, 2)


def compact_blocks_for_labeling(pdf_blocks: list[dict]) -> list[dict]:
    pdf_blocks = clean_blocks(pdf_blocks)
    body_font_size = get_body_font_size(pdf_blocks)

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
        font_size = get_font_size(block)

        x0 = get_layout_value(block, "x0")
        x1 = get_layout_value(block, "x1")
        top = get_layout_value(block, "top")
        bottom = get_layout_value(block, "bottom")

        compact_blocks.append(
            {
                "block_id": i + 1,
                "text": text,
                "page": block.get("page"),
                "line_order": block.get("line_order"),
                "position_hint": get_position_hint(
                    index=i,
                    total_blocks=total_blocks,
                    page=block.get("page"),
                ),
                "font_size": font_size,
                "body_font_size": body_font_size,
                "relative_font": get_relative_font_hint(
                    font_size,
                    body_font_size,
                ),
                "is_bold": is_bold_block(block),
                "visual_hint": get_visual_hint(block, body_font_size),
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
                "indent": x0,
                "vertical_gap_before": get_vertical_gap_before(
                    clean_pdf_blocks,
                    i,
                ),
                "vertical_gap_after": get_vertical_gap_after(
                    clean_pdf_blocks,
                    i,
                ),
                "word_count": len(text.split()),
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

Use PDFPlumber text and layout metadata:
- text
- page
- line_order
- position_hint
- font_size
- body_font_size
- relative_font
- is_bold
- visual_hint
- x0
- x1
- top
- bottom
- indent
- vertical_gap_before
- vertical_gap_after
- word_count

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
"Element 1: Data Type:"
"2. Data used in publications"
"3. "TYPES OF DATA"
"3. Policies for access and sharing."
"5. Plans for archiving and preservation of access."

subsection:
A smaller prompt, question, or internal heading that belongs under a major section.
Examples may include numbered headings such as:
"A. Types and amount of scientific data expected to be generated in the project:"
"B. Scientific data that will be preserved and shared, and the rationale for doing so:"
"C. Metadata, other relevant data, and associated documentation:"
"Data management plans should describe whether and how data generated in the course of the proposed research will be shared and preserved. If the plan is not to share and/or preserve certain data, then the plan must explain the basis of the decision (for example, cost/benefit considerations, other parameters of feasibility, scientific appropriateness, or limitations discussed in #4). At a minimum, DMPs must describe how data sharing and preservation will enable validation of results, or how results could be validated if data are not shared or preserved."

content:
Narrative body text, answers, explanations, guidance, examples, list items,
data descriptions, repository names, institution names, continuation lines,
and anything that is not clearly a title, section, or subsection.

Important instructions:
- Return ONLY valid JSON.
- Do NOT rewrite text.
- Do NOT summarize text.
- Do NOT invent headings.
- Return labels ONLY for target_block_ids.
- Do NOT return labels for context-only blocks.
- Do NOT omit any target_block_id.
- Do NOT add new block_ids.
- Do NOT change block order.
- If uncertain, label the block as content.

Return format:
[
  {{"block_id": 1, "label": "document_title"}},
  {{"block_id": 2, "label": "section"}},
  {{"block_id": 3, "label": "content"}}
]

Chunk blocks:
{blocks_text}
"""



def extract_json_array(model_output: str) -> str:
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


def build_labeling_diagnostics(
    compact_blocks: list[dict],
    label_lookup: dict[int, str],
    chunk_diagnostics: list[dict] | None = None,
) -> dict:
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


def postprocess_blocks(blocks: list[dict]) -> list[dict]:
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


def label_blocks_with_chunked_llm(
    llm,
    compact_blocks: list[dict],
    chunk_size: int = 45,
    overlap: int = 5,
) -> tuple[dict[int, str], list[dict]]:
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
    return generate_structured_blocks_with_llm_labels_only(
        llm=llm,
        pdf_blocks=pdf_blocks,
        return_diagnostics=return_diagnostics,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def save_blocks(blocks: list[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)


def save_diagnostics(diagnostics: dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)