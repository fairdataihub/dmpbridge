# src/dmpbridge/llm/llm_narrative_blocks.py

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


# ============================================================
# Rule-based structure detection
# ============================================================

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

    if not sizes:
        return 11.0

    return median(sizes)


def is_bold_block(block: dict) -> bool:
    if block.get("is_bold") is True:
        return True

    font_name = str(block.get("font_name", "")).lower()

    bold_markers = [
        "bold",
        "black",
        "heavy",
        "semibold",
        "demibold",
    ]

    return any(marker in font_name for marker in bold_markers)


def normalize_text_simple(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", text.strip()))

def is_punctuation_only(text: str) -> bool:
    return bool(re.match(r"^[^\w]+$", text.strip()))

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

    # Example:
    # 1. Policy and Practice. For the proposed research...
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

    # Avoid single-word uppercase subtitles like ELECTRODES
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
    """
    Rule-based PDFPlumber structure detector.

    Labels:
    - document_title
    - section
    - subsection
    - content
    - subtitle
    - empty

    This function uses:
    - text patterns
    - numbering
    - NIH Element headings
    - all-caps headings
    - relative font size
    - bold formatting
    """

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


# ============================================================
# LLM prompt
# ============================================================
def build_llm_blocks_prompt(blocks_text: str) -> str:
    return f"""
You are helping extract the existing narrative structure from a Data Management Plan.

Your task is EXTRACTION ONLY.

Allowed labels:
- document_title
- section
- subsection
- content

Definitions:
- document_title: the main title of the DMP.
- section: an existing major heading copied exactly from the DMP.
- subsection: an existing sub-question or prompt copied exactly from the DMP.
- content: body text, instruction text, guidance text, answer text, explanation, or paragraph text.

Core extraction rules:
1. Do not omit content blocks. 
2. Do not invent text.
3. Do not infer hidden headings.
4. Do not create new headings.
5. Do not summarize text into headings.
6. Do not rename headings.
7. Do not use your own wording.
8. A section or subsection must appear explicitly in the input blocks.
9. Do not include page numbers.
10. Return only the block array.
11. Examples are illustrative only. Never copy example text unless it appears in the current input blocks.

You are given PDFPlumber extracted blocks.
Each block may include:
- block_id
- text
- page
- font_size
- is_bold

Use both the text and available formatting information to decide the label.

How to detect document_title:
Use "document_title" only for the main DMP title, usually near the beginning of the document.

How to detect sections:
Use "section" only for existing major DMP headings.

A section is usually:
- a numbered heading, such as "1. Policy and Practice"
- an element heading, such as "Element 1: Data Type:"
- a short standalone heading line
- a heading followed by multiple paragraphs of related content
- a heading that introduces a major DMP topic
- a block that appears visually like a heading, for example larger font or bold, if that information is available


How to detect subsections:
Use "subsection" only for existing prompts inside a section.

A subsection is usually:
- a lettered prompt, such as "A. Types and amount of scientific data expected to be generated in the project:"
- a short prompt ending with a colon or period inside a larger numbered section
- a repeated internal prompt under a major heading
- a block that appears visually like a smaller heading or prompt under a section

How to detect content:
Use "content" for:
- full sentences
- paragraphs
- guidance or instruction text
- answer text
- explanatory text
- body text after a heading
- body text after a prompt

Important negative rules:
- A paragraph should never become a section.
- A paragraph should never become a subsection.

Inline heading rule:
A short phrase at the beginning of a block followed by a period may be a subsection if it is heading-like and followed by substantial explanatory content.
Inline heading rule:
If a real heading and its content appear in the same block, split them into two blocks:
1. section or subsection
2. content


Examples:
"Roles & Responsibilities. For the proposed research..."
"Data Types and Sources. The proposed research..."
"Content and Format. A statement of plans..."


Example:
"1. Types of data. The bulk of the data generated in this project will be..."
should become:
[
  {{
    "label": "section",
    "text": "1. Types of data."
  }},
  {{
    "label": "content",
    "text": "The bulk of the data generated in this project will be..."
  }}
]

Example:
"Roles & Responsibilities. For the proposed research, Director Samuel Stupp..."
should become:
[
  {{
    "label": "subsection",
    "text": "Roles & Responsibilities."
  }},
  {{
    "label": "content",
    "text": "For the proposed research, Director Samuel Stupp..."
  }}
]

Return format:
[
  {{
    "label": "document_title",
    "text": "..."
  }},
  {{
    "label": "section",
    "text": "..."
  }},
  {{
    "label": "subsection",
    "text": "..."
  }},
  {{
    "label": "content",
    "text": "..."
  }}
]

PDFPlumber blocks:
{blocks_text}
"""


# ============================================================
# JSON extraction and post-processing
# ============================================================

def extract_json_array(model_output: str) -> str:
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    if label in ALLOWED_LABELS:
        return label

    return ""


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
        {
            "label": "section",
            "text": section_text,
        },
        {
            "label": "content",
            "text": content_text,
        },
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


def compact_pdfplumber_blocks(pdf_blocks: list[dict]) -> list[dict]:
    compact_blocks = []

    for i, block in enumerate(pdf_blocks, start=1):
        text = normalize_text_simple(block.get("text", ""))

        if not text:
            continue

        compact_blocks.append(
            {
                "block_id": i,
                "text": text,
                "page": block.get("page"),
                "font_size": block.get("font_size"),
                "avg_font_size": block.get("avg_font_size"),
                "is_bold": block.get("is_bold"),
                "label": block.get("label"),
            }
        )

    return compact_blocks


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

        # IMPORTANT FIX:
        # Only split dot-style subsections when the block is already subsection.
        # Do NOT run this on content, because normal content sentences may end with a period.
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

        clean_output.append(
            {
                "label": label,
                "text": text,
            }
        )

    return clean_output

def merge_consecutive_content_blocks(blocks):
    merged = []

    for block in blocks:

        if (
            merged
            and block["label"] == "content"
            and merged[-1]["label"] == "content"
        ):

            prev = merged[-1]["text"]

            if prev.endswith((".", ":", ";", "?", "!")):
                merged[-1]["text"] += "\n\n" + block["text"]
            else:
                merged[-1]["text"] += " " + block["text"]

        else:
            merged.append(dict(block))

    return merged

# ============================================================
# Main functions
# ============================================================

def generate_structured_blocks_with_rules(pdf_blocks: list[dict]) -> list[dict]:
    """
    Use only rule-based detection.
    Good for debugging PDFPlumber extraction.
    """

    structured = detect_structure(pdf_blocks)

    simplified = [
        {
            "label": block["label"],
            "text": block["text"],
        }
        for block in structured
        if block.get("label") not in {"empty", "subtitle"}
    ]

    blocks = postprocess_blocks(simplified)
    blocks = merge_consecutive_content_blocks(blocks)

    return blocks


def generate_structured_blocks_with_llm(llm, pdf_blocks: list[dict]) -> list[dict]:
    """
    Hybrid method:
    1. Run rule-based detection first.
    2. Send labeled blocks to LLM.
    3. Post-process LLM output safely.
    4. Merge consecutive content blocks.
    """

    rule_labeled_blocks = detect_structure(pdf_blocks)
    compact_blocks = compact_pdfplumber_blocks(rule_labeled_blocks)

    blocks_text = json.dumps(
        compact_blocks,
        indent=2,
        ensure_ascii=False,
    )

    prompt = build_llm_blocks_prompt(blocks_text)

    response = llm.invoke(prompt)
    model_text = response.content

    json_text = extract_json_array(model_text)

    try:
        blocks = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        blocks = json.loads(repaired)

    blocks = postprocess_blocks(blocks)
    blocks = merge_consecutive_content_blocks(blocks)

    return blocks


def save_blocks(blocks: list[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)