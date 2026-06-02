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



# Basic utilities


def normalize_text_simple(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", text.strip()))


def is_punctuation_only(text: str) -> bool:
    return bool(re.match(r"^[^\w]+$", text.strip()))


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def get_font_size(block: dict) -> float:
    return (
        block.get("avg_font_size")
        or block.get("font_size")
        or block.get("size")
        or 0
    )


def get_body_font_size(blocks: list[dict]) -> float:
    sizes = [
        get_font_size(block)
        for block in blocks
        if get_font_size(block) > 0 and block.get("text", "").strip()
    ]

    return median(sizes) if sizes else 11.0


def is_bold_block(block: dict) -> bool:
    if block.get("is_bold") is True:
        return True

    font_names = block.get("font_names") or []
    font_name = str(block.get("font_name", "")).lower()
    all_fonts = " ".join(font_names).lower() + " " + font_name

    bold_markers = ["bold", "black", "heavy", "semibold", "demibold"]

    return any(marker in all_fonts for marker in bold_markers)



# Rule-based detection

def is_document_title_text(text: str) -> bool:
    text = normalize_text_simple(text)

    known_titles = [
        r"^DATA MANAGEMENT AND SHARING PLAN$",
        r"^DATA MANAGEMENT PLAN$",
        r"^DATA MANAGEMENT$",
        r"^DMP$",
        r"^RESOURCE/DATA SHARING PLAN$",
    ]

    return any(
        re.match(pattern, text, flags=re.IGNORECASE)
        for pattern in known_titles
    )


def is_nih_element_heading(text: str) -> bool:
    return bool(
        re.match(
            r"^Element\s+\d+\s*:",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def looks_like_sentence(text: str) -> bool:
    words = [
        word.strip(".,;:()[]{}").lower()
        for word in text.split()
    ]

    sentence_markers = {
        "we", "our", "this", "these", "those",
        "is", "are", "was", "were", "will", "would",
        "should", "could", "can", "may", "must",
        "be", "been", "being", "have", "has", "had",
    }

    return len(words) >= 7 and any(word in sentence_markers for word in words)


def is_numbered_section(text: str) -> bool:
    text = normalize_text_simple(text)

    if not text:
        return False

    inline_match = re.match(r"^\d+\.\s+(.+?)\.\s+.+", text)

    if inline_match:
        heading_part = inline_match.group(1).strip()
        words = heading_part.split()
        return 2 <= len(words) <= 12

    words = text.split()

    if len(words) > 14:
        return False

    return bool(re.match(r"^\d+\.\s+[A-Z][A-Za-z]", text))


def is_decimal_subsection(text: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\s+", text.strip()))


def is_lettered_subsection(text: str) -> bool:
    return bool(re.match(r"^[A-Z]\.\s+.+", text.strip()))


def is_all_caps_heading(text: str) -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if not (2 <= len(words) <= 12):
        return False

    if looks_like_sentence(text):
        return False

    return text.isupper()


def is_short_uppercase_subtitle(text: str) -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if not (1 <= len(words) <= 5):
        return False

    return text.isupper()


def is_prompt_style_subsection(text: str) -> bool:
    text = normalize_text_simple(text)

    if not text.endswith(":"):
        return False

    words = text.split()

    if len(words) > 12:
        return False

    if looks_like_sentence(text):
        return False

    return True


def is_title_style_section(block: dict, body_font_size: float) -> bool:
    text = normalize_text_simple(block.get("text", ""))
    font_size = get_font_size(block)
    is_bold = is_bold_block(block)

    if not text:
        return False

    words = text.split()

    if not (2 <= len(words) <= 12):
        return False

    if text.endswith("?"):
        return False

    if looks_like_sentence(text):
        return False

    visually_heading = (
        font_size >= body_font_size + 1
        or font_size >= body_font_size * 1.12
        or is_bold
    )

    return visually_heading


def is_false_section_fragment(text: str) -> bool:
    """
    Demote wrapped sentence fragments incorrectly labeled as section.
    Useful for sample2/sample3 where bold guidance text wraps across lines.
    """

    text = normalize_text_simple(text)
    words = text.split()

    if not text:
        return False

    # Keep strong real headings
    if is_nih_element_heading(text):
        return False

    if re.match(r"^\d+\.\s+[A-Z]", text):
        return False

    if is_all_caps_heading(text):
        return False

    if is_lettered_subsection(text):
        return False

    # Wrapped fragments usually start lowercase
    if text[0].islower():
        return True

    # Long sentence-like text should not be a section
    if looks_like_sentence(text):
        return True

    # Guidance fragments with comma/semicolon are usually content
    if any(p in text for p in [";", ","]) and len(words) >= 5:
        return True

    # Short ending fragments like "maintained during the project."
    if text.endswith(".") and len(words) <= 6:
        return True

    return False


def is_same_as_document_title(text: str, document_title: str | None) -> bool:
    if not document_title:
        return False

    return (
        normalize_text_simple(text).lower()
        == normalize_text_simple(document_title).lower()
    )


def classify_line(block: dict, body_font_size: float) -> str:
    text = normalize_text_simple(block.get("text", ""))

    if not text:
        return "empty"

    if is_page_number(text):
        return "empty"

    if is_document_title_text(text):
        return "document_title"

    if is_nih_element_heading(text):
        return "section"

    if is_numbered_section(text):
        return "section"

    if is_all_caps_heading(text):
        return "section"

    if is_title_style_section(block, body_font_size):
        return "section"

    if is_lettered_subsection(text):
        return "subsection"

    if is_decimal_subsection(text):
        return "subsection"

    if is_prompt_style_subsection(text):
        return "subsection"

    return "content"


def detect_structure(blocks: list[dict]) -> list[dict]:
    blocks = clean_blocks(blocks)
    body_font_size = get_body_font_size(blocks)

    structured_blocks = []
    seen_document_title = False
    seen_first_section = False
    document_title = None

    for block in blocks:
        text = normalize_text_simple(block.get("text", ""))

        if not text or is_page_number(text):
            label = "empty"

        elif not seen_document_title:
            label = "document_title"
            document_title = text
            seen_document_title = True

        elif not seen_first_section and is_short_uppercase_subtitle(text):
            label = "subtitle"

        elif is_same_as_document_title(text, document_title):
            label = "subtitle"

        else:
            label = classify_line(block, body_font_size)

            if label == "section":
                seen_first_section = True

        structured_blocks.append(
            {
                **block,
                "text": text,
                "label": label,
            }
        )

    return structured_blocks



# LLM label-only prompt


def build_llm_label_only_prompt(blocks_text: str) -> str:
    return f"""
You are labeling PDF-extracted Data Management Plan blocks.

IMPORTANT:
Return ONLY labels for existing block_ids.
Do NOT return text.
Do NOT rewrite text.
Do NOT summarize text.
Do NOT omit any block_id.
Every input block_id must appear exactly once.

Allowed labels:
- document_title
- section
- subsection
- content

Rules:
1. document_title = main DMP title, usually first block.
2. section = major heading, numbered heading, Element heading, or major visual heading.
3. subsection = smaller prompt under a section, lettered prompt, decimal prompt, or short colon-ended prompt.
4. content = body text, paragraph text, explanations, instructions, guidance, or answers.
5. If unsure, use content.
6. Do not label repository names as subsection.
7. Do not label institution names as subsection.
8. Do not label continuation lines as subsection.
9. Do not label short fragments like "California San Diego Library repository." as subsection.
10. Do not label short fragments like "Coordinating Center." as subsection.
11. Do not label wrapped sentence fragments as section.
12. Do not return page numbers.


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
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def compact_blocks_for_labeling(pdf_blocks: list[dict]) -> list[dict]:
    compact_blocks = []

    for i, block in enumerate(pdf_blocks, start=1):
        text = normalize_text_simple(block.get("text", ""))

        if not text or is_page_number(text):
            continue

        compact_blocks.append(
            {
                "block_id": i,
                "text": text,
                "page": block.get("page"),
                "avg_font_size": block.get("avg_font_size"),
                "font_size": block.get("font_size"),
                "is_bold": block.get("is_bold"),
                "rule_label": block.get("label"),
            }
        )

    return compact_blocks



# Post-processing


def split_inline_numbered_section(text: str) -> list[dict]:
    text = normalize_text_simple(text)

    pattern = r"^(\d+\.\s+[^.]+\.)\s+(.+)$"
    match = re.match(pattern, text)

    if not match:
        return []

    section_text = match.group(1).strip()
    content_text = match.group(2).strip()

    if not section_text or not content_text:
        return []

    return [
        {"label": "section", "text": section_text},
        {"label": "content", "text": content_text},
    ]


def split_inline_dot_subsection(text: str) -> list[dict]:
    text = normalize_text_simple(text)

    pattern = r"^([A-Z][A-Za-z0-9/&,\-\(\),\s]{2,80}\.)\s+(.+)$"
    match = re.match(pattern, text)

    if not match:
        return []

    subsection_text = match.group(1).strip()
    content_text = match.group(2).strip()

    heading_words = subsection_text.rstrip(".").split()

    if not (2 <= len(heading_words) <= 8):
        return []

    if len(content_text.split()) < 5:
        return []

    if looks_like_sentence(subsection_text):
        return []

    return [
        {"label": "subsection", "text": subsection_text},
        {"label": "content", "text": content_text},
    ]


def is_sentence_or_paragraph(text: str) -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if len(words) > 14:
        return True

    starters = (
        "The ",
        "This ",
        "These ",
        "All ",
        "We ",
        "Data ",
        "Materials ",
        "Interested ",
        "Local ",
        "Stored ",
        "If ",
        "As ",
        "In ",
        "Additionally,",
        "Researchers ",
        "Confidential ",
        "Software ",
        "Select ",
        "Research ",
        "Upon ",
        "Our ",
        "First,",
        "Specifically,",
    )

    return text.startswith(starters)


def postprocess_blocks(blocks: list[dict]) -> list[dict]:
    clean_output = []
    document_title_used = False

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

        if label == "document_title":
            if document_title_used:
                label = "content"
            else:
                document_title_used = True

        if label in {"section", "content"}:
            split_blocks = split_inline_numbered_section(text)

            if split_blocks:
                clean_output.extend(split_blocks)
                continue

        if label == "subsection":
            split_blocks = split_inline_dot_subsection(text)

            if split_blocks:
                clean_output.extend(split_blocks)
                continue

        if is_nih_element_heading(text):
            label = "section"

        elif is_numbered_section(text):
            label = "section"

        elif is_lettered_subsection(text):
            label = "subsection"

        elif label == "subsection" and is_sentence_or_paragraph(text):
            label = "content"

        elif label == "section" and is_sentence_or_paragraph(text):
            label = "content"

        if label == "subsection":
            words = text.rstrip(".:").split()

            if len(words) < 3 and not is_lettered_subsection(text):
                label = "content"

        if label == "section" and is_false_section_fragment(text):
            label = "content"

        clean_output.append(
            {
                "label": label,
                "text": text,
            }
        )

    return clean_output


def merge_consecutive_content_blocks(blocks: list[dict]) -> list[dict]:
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


# ============================================================
# Main functions
# ============================================================

def generate_structured_blocks_with_rules(pdf_blocks: list[dict]) -> list[dict]:
    structured = detect_structure(pdf_blocks)

    simplified = [
        {
            "label": block["label"],
            "text": block["text"],
        }
        for block in structured
        if block.get("label") not in {"empty"}
    ]

    blocks = postprocess_blocks(simplified)
    blocks = merge_consecutive_content_blocks(blocks)

    return blocks


def generate_structured_blocks_with_llm_labels_only(
    llm,
    pdf_blocks: list[dict],
) -> list[dict]:
    rule_labeled_blocks = detect_structure(pdf_blocks)
    compact_blocks = compact_blocks_for_labeling(rule_labeled_blocks)

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

    json_text = extract_json_array(model_text)

    try:
        llm_labels = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        llm_labels = json.loads(repaired)

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
        text = normalize_text_simple(original_block.get("text", ""))

        if not text or is_page_number(text) or is_punctuation_only(text):
            continue

        rule_label = normalize_label(original_block.get("rule_label", ""))

        label = label_lookup.get(
            block_id,
            rule_label or "content",
        )

        text = clean_repeated_words(text)

        rebuilt_blocks.append(
            {
                "label": label,
                "text": text,
            }
        )

    blocks = postprocess_blocks(rebuilt_blocks)
    blocks = merge_consecutive_content_blocks(blocks)

    return blocks


def generate_structured_blocks_with_llm(
    llm,
    pdf_blocks: list[dict],
) -> list[dict]:
    return generate_structured_blocks_with_llm_labels_only(
        llm=llm,
        pdf_blocks=pdf_blocks,
    )


def save_blocks(blocks: list[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)