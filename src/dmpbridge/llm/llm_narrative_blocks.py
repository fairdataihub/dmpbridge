"""
llm_narrative_blocks.py

Purpose:
Convert PDFPlumber-extracted DMP blocks into structured narrative blocks using an LLM-only labeling approach.

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


def normalize_text_simple(text: str) -> str:
    """Normalize whitespace while preserving the original wording."""
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_label(label: str) -> str:
    """Keep only labels allowed by the final narrative schema."""
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def is_page_number(text: str) -> bool:
    """Remove standalone page numbers from the final output."""
    return bool(re.match(r"^\d+$", text.strip()))


def is_punctuation_only(text: str) -> bool:
    """Remove blocks that contain only punctuation or symbols."""
    return bool(re.match(r"^[^\w]+$", text.strip()))


def get_font_size(block: dict) -> float:
    """Read font size from possible PDFPlumber/debug block fields."""
    return (
        block.get("avg_font_size")
        or block.get("font_size")
        or block.get("size")
        or 0
    )


def get_body_font_size(blocks: list[dict]) -> float:
    """Estimate the normal body font size using the median font size."""
    sizes = [
        get_font_size(block)
        for block in blocks
        if get_font_size(block) > 0 and block.get("text", "").strip()
    ]

    return median(sizes) if sizes else 11.0


def is_bold_block(block: dict) -> bool:
    """Detect bold style from explicit metadata or font names."""
    if block.get("is_bold") is True:
        return True

    font_names = block.get("font_names") or []
    font_name = str(block.get("font_name", "")).lower()
    all_fonts = " ".join(font_names).lower() + " " + font_name

    return any(
        marker in all_fonts
        for marker in ["bold", "black", "heavy", "semibold", "demibold"]
    )


def get_visual_hint(block: dict, body_font_size: float) -> str:
    """
    Create a simple visual hint for the LLM.

    This is not used to assign labels directly. It only gives the LLM layout context.
    """
    text = normalize_text_simple(block.get("text", ""))
    font_size = get_font_size(block)

    if is_bold_block(block):
        return "bold"

    if font_size >= body_font_size + 1 or font_size >= body_font_size * 1.12:
        return "larger_than_body"

    if text.isupper() and len(text.split()) <= 18:
        return "uppercase_text"

    return "body_text"


def compact_blocks_for_labeling(pdf_blocks: list[dict]) -> list[dict]:
    """
    Convert raw PDF blocks into compact LLM input.

    Each item includes surrounding text and visual metadata.
    No rule-based labels are generated or passed to the model.
    """
    pdf_blocks = clean_blocks(pdf_blocks)
    body_font_size = get_body_font_size(pdf_blocks)

    clean_pdf_blocks = []

    for block in pdf_blocks:
        text = normalize_text_simple(block.get("text", ""))

        if not text:
            continue

        if is_page_number(text):
            continue

        if is_punctuation_only(text):
            continue

        clean_pdf_blocks.append(block)

    compact_blocks = []

    for i, block in enumerate(clean_pdf_blocks):
        current_text = normalize_text_simple(block.get("text", ""))

        previous_text = ""
        next_text = ""

        if i > 0:
            previous_text = normalize_text_simple(
                clean_pdf_blocks[i - 1].get("text", "")
            )

        if i + 1 < len(clean_pdf_blocks):
            next_text = normalize_text_simple(
                clean_pdf_blocks[i + 1].get("text", "")
            )

        compact_blocks.append(
            {
                "block_id": i + 1,
                "previous_text": previous_text,
                "current_text": current_text,
                "next_text": next_text,
                "page": block.get("page"),
                "line_order": block.get("line_order"),
                "font_size": get_font_size(block),
                "body_font_size": body_font_size,
                "is_bold": is_bold_block(block),
                "visual_hint": get_visual_hint(block, body_font_size),
            }
        )

    return compact_blocks


def build_llm_label_only_prompt(blocks_text: str) -> str:
    return f"""
You are an expert at reading Data Management Plans.

Your task is to classify each CURRENT block into one narrative label.

Use only:
- previous_text
- current_text
- next_text
- page
- line_order
- font_size
- body_font_size
- is_bold
- visual_hint

Return ONLY valid JSON.
Do NOT rewrite text.
Do NOT summarize text.
Do NOT invent headings.
Do NOT omit any block_id.
Do NOT add extra block_ids.

Allowed labels:
- document_title
- section
- subsection
- content

Definitions:

document_title:
The main title of the Data Management Plan. It usually appears near the beginning of page 1. It may span multiple consecutive beginning blocks.

section:
A major heading that starts a new DMP topic or major requirement area.

subsection:
A smaller prompt, question, or internal heading inside a section.

content:
Paragraphs, answer text, explanatory text, guidance text, list items, examples, data descriptions, repository names, institution names, and wrapped continuation lines.

Decision guidance:
1. Decide whether current_text functions as a title, major heading, smaller heading, or body content.
2. Use previous_text and next_text to detect whether current_text continues a sentence or starts a new topic.
3. Use visual metadata only as context. Do not rely on visual metadata alone.
4. If current_text is part of a paragraph, explanation, list, example, or continuation, label it as content.
5. If current_text is the main DMP title at the beginning, label it as document_title.
6. If current_text opens a major DMP topic, label it as section.
7. If current_text is a smaller question or prompt within a major topic, label it as subsection.
8. When uncertain between heading and content, prefer content.

Return format:
[
  {{"block_id": 1, "label": "document_title"}},
  {{"block_id": 2, "label": "section"}},
  {{"block_id": 3, "label": "content"}}
]

Input blocks:
{blocks_text}
"""


def extract_json_array(model_output: str) -> str:
    """Extract the JSON array from the model response."""
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def parse_llm_labels(model_text: str) -> list[dict]:
    """Parse LLM JSON and repair minor JSON formatting problems if needed."""
    json_text = extract_json_array(model_text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        return json.loads(repaired)


def postprocess_blocks(blocks: list[dict]) -> list[dict]:
    """
    Light cleanup after LLM labeling.

    It only validates labels, removes empty/noisy blocks, and cleans repeated words.
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
    """Merge title blocks when the title is split across consecutive lines."""
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
    """Merge neighboring content blocks to improve paragraph readability."""
    merged = []

    for block in blocks:
        if (
            merged
            and block["label"] == "content"
            and merged[-1]["label"] == "content"
        ):
            prev = merged[-1]["text"]
            current = block["text"]

            if prev.endswith("-"):
                merged[-1]["text"] = prev[:-1] + current

            elif prev.endswith((".", ":", ";", "?", "!")):
                merged[-1]["text"] += "\n\n" + current

            else:
                merged[-1]["text"] += " " + current

        else:
            merged.append(dict(block))

    return merged


def generate_structured_blocks_with_llm_labels_only(
    llm,
    pdf_blocks: list[dict],
) -> list[dict]:
    """Generate narrative blocks using only LLM labels."""
    compact_blocks = compact_blocks_for_labeling(pdf_blocks)

    block_lookup = {
        block["block_id"]: block
        for block in compact_blocks
    }

    blocks_text = json.dumps(
        compact_blocks,
        indent=2,
        ensure_ascii=False,
    )

    prompt = build_llm_label_only_prompt(blocks_text)

    response = llm.invoke(prompt)
    model_text = response.content

    llm_labels = parse_llm_labels(model_text)

    label_lookup = {}

    for item in llm_labels:
        if not isinstance(item, dict):
            continue

        block_id = item.get("block_id")
        label = normalize_label(item.get("label", ""))

        if block_id in block_lookup and label:
            label_lookup[block_id] = label

    rebuilt_blocks = []

    for block_id, original_block in block_lookup.items():
        text = normalize_text_simple(original_block.get("current_text", ""))

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

    return blocks


def generate_structured_blocks_with_llm(
    llm,
    pdf_blocks: list[dict],
) -> list[dict]:
    """Main public function for LLM-only narrative block generation."""
    return generate_structured_blocks_with_llm_labels_only(
        llm=llm,
        pdf_blocks=pdf_blocks,
    )


def save_blocks(blocks: list[dict], output_path):
    """Save structured narrative blocks as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)
