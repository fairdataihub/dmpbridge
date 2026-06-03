"""
llm_narrative_blocks.py

Purpose:
Convert PDFPlumber-extracted DMP blocks into structured narrative blocks.

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

    return any(
        marker in all_fonts
        for marker in ["bold", "black", "heavy", "semibold", "demibold"]
    )


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
    text = normalize_text_simple(text)
    return re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE) is not None


def looks_like_sentence(text: str) -> bool:
    text = normalize_text_simple(text)

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


def starts_like_sentence(text: str) -> bool:
    text = normalize_text_simple(text)

    starters = (
        "The ", "This ", "These ", "Those ", "We ", "Our ",
        "All ", "Data ", "Materials ", "Researchers ",
        "In ", "As ", "If ", "When ", "Because ",
        "Additionally,", "Furthermore,", "However,"
    )

    return text.startswith(starters)


def is_probable_paragraph(text: str) -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if not text:
        return False

    if len(words) >= 15:
        return True

    if text.endswith(".") and len(words) >= 8:
        return True

    if starts_like_sentence(text) and len(words) >= 7:
        return True

    if looks_like_sentence(text):
        return True

    return False


def is_list_item(text: str) -> bool:
    text = normalize_text_simple(text)

    return bool(
        re.match(r"^\([ivxlcdm]+\)\s+", text, flags=re.IGNORECASE)
        or re.match(r"^\([a-z]\)\s+", text, flags=re.IGNORECASE)
        or re.match(r"^\d+[\).]\s+", text)
        or re.match(r"^[-•]\s+", text)
    )


def is_metric_or_example_line(text: str) -> bool:
    text = normalize_text_simple(text)

    if "–" in text or " - " in text:
        if not text.endswith(":"):
            return True

    return False


def is_wrapped_continuation_line(text: str, prev_text: str = "") -> bool:
    text = normalize_text_simple(text)
    prev_text = normalize_text_simple(prev_text)

    if not text:
        return False

    if prev_text.endswith((",", "and", "or", "of", "to", "in", "at")):
        return True

    words = text.split()

    if len(words) >= 8 and not text.endswith(":"):
        sentence_words = {
            "that", "which", "will", "should", "must",
            "are", "is", "be", "been", "being",
            "to", "of", "at", "with", "for", "from",
        }

        lower_words = {
            word.strip(".,;:").lower()
            for word in words
        }

        if lower_words & sentence_words:
            return True

    return False


def is_inline_prompt_heading(text: str) -> bool:
    text = normalize_text_simple(text)

    pattern = r"^([A-Z][A-Za-z&/\-,\s]{2,60})\.\s+(.+)$"
    match = re.match(pattern, text)

    if not match:
        return False

    heading_part = match.group(1).strip()
    rest = match.group(2).strip()

    if len(heading_part.split()) > 6:
        return False

    if len(rest.split()) < 4:
        return False

    return True


def is_numbered_section(text: str) -> bool:
    text = normalize_text_simple(text)

    if not text:
        return False

    inline_match = re.match(r"^\d+\.\s+(.+?)\.\s+.+", text)

    if inline_match:
        heading_part = inline_match.group(1).strip()
        words = heading_part.split()
        return 2 <= len(words) <= 14

    words = text.split()

    if len(words) > 16:
        return False

    return bool(re.match(r"^\d+\.\s+[A-Z][A-Za-z]", text))


def is_decimal_subsection(text: str) -> bool:
    text = normalize_text_simple(text)
    return bool(re.match(r"^\d+\.\d+\s+", text))


def is_lettered_subsection(text: str) -> bool:
    text = normalize_text_simple(text)
    return bool(re.match(r"^[A-Z]\.\s+.+", text))


def is_all_caps_heading(text: str) -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if not (2 <= len(words) <= 16):
        return False

    if is_probable_paragraph(text):
        return False

    return text.isupper()


def is_prompt_style_subsection(text: str) -> bool:
    text = normalize_text_simple(text)

    if not text.endswith(":"):
        return False

    words = text.split()

    if len(words) > 16:
        return False

    if is_probable_paragraph(text):
        return False

    return True


def is_title_style_section(block: dict, body_font_size: float) -> bool:
    text = normalize_text_simple(block.get("text", ""))
    font_size = get_font_size(block)

    if not text:
        return False

    words = text.split()

    if not (2 <= len(words) <= 16):
        return False

    if text.endswith("?"):
        return False

    if is_list_item(text):
        return False

    if is_metric_or_example_line(text):
        return False

    if is_probable_paragraph(text):
        return False

    return (
        font_size >= body_font_size + 1
        or font_size >= body_font_size * 1.12
        or is_bold_block(block)
    )


def is_standalone_heading_candidate(text: str) -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if not text:
        return False

    if is_list_item(text):
        return False

    if is_metric_or_example_line(text):
        return False

    if len(words) < 2:
        return False

    if len(words) > 18:
        return False

    if text.endswith((".", ";")):
        return False

    if is_probable_paragraph(text):
        return False

    if text[0].islower():
        return False

    return True


def is_context_section(
    blocks: list[dict],
    index: int,
    body_font_size: float,
) -> bool:
    block = blocks[index]
    text = normalize_text_simple(block.get("text", ""))

    if not is_standalone_heading_candidate(text):
        return False

    font_size = get_font_size(block)

    next_text = ""
    if index + 1 < len(blocks):
        next_text = normalize_text_simple(blocks[index + 1].get("text", ""))

    prev_text = ""
    if index > 0:
        prev_text = normalize_text_simple(blocks[index - 1].get("text", ""))

    next_is_paragraph = is_probable_paragraph(next_text)
    prev_is_paragraph = is_probable_paragraph(prev_text)

    visual_heading = (
        is_bold_block(block)
        or font_size >= body_font_size + 1
        or font_size >= body_font_size * 1.12
    )

    if next_is_paragraph:
        return True

    if visual_heading:
        return True

    if prev_is_paragraph and is_standalone_heading_candidate(text):
        return True

    return False


def is_top_title_candidate(blocks: list[dict], index: int) -> bool:
    block = blocks[index]
    text = normalize_text_simple(block.get("text", ""))
    words = text.split()

    if not text:
        return False

    page = block.get("page", 1)

    if page not in {1, "1", None}:
        return False

    if index > 2:
        return False

    if len(words) > 18:
        return False

    if is_probable_paragraph(text):
        return False

    if text.isupper():
        return True

    if is_document_title_text(text):
        return True

    uppercase_words = sum(1 for word in words if word[:1].isupper())

    return uppercase_words >= max(1, len(words) // 2)


def is_false_section_fragment(text: str, prev_text: str = "") -> bool:
    text = normalize_text_simple(text)
    words = text.split()

    if not text:
        return False

    if is_wrapped_continuation_line(text, prev_text):
        return True

    if is_list_item(text):
        return True

    if is_metric_or_example_line(text):
        return True

    if is_nih_element_heading(text):
        return False

    if re.match(r"^\d+\.\s+[A-Z]", text):
        return False

    if is_all_caps_heading(text):
        return False

    if is_lettered_subsection(text):
        return False

    if text[0].islower():
        return True

    if is_probable_paragraph(text):
        return True

    if any(p in text for p in [";", ","]) and len(words) >= 6:
        return True

    if text.endswith(".") and len(words) <= 6:
        return True

    return False


def classify_line(
    block: dict,
    body_font_size: float,
    blocks: list[dict] | None = None,
    index: int | None = None,
) -> str:
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

    if is_decimal_subsection(text):
        return "subsection"

    if is_lettered_subsection(text):
        return "subsection"

    if is_prompt_style_subsection(text):
        return "subsection"

    if is_all_caps_heading(text):
        return "section"

    if is_title_style_section(block, body_font_size):
        return "section"

    if blocks is not None and index is not None:
        if is_context_section(blocks, index, body_font_size):
            return "section"

    return "content"


def detect_structure(blocks: list[dict]) -> list[dict]:
    blocks = clean_blocks(blocks)
    body_font_size = get_body_font_size(blocks)

    structured_blocks = []
    seen_first_section = False

    for i, block in enumerate(blocks):
        text = normalize_text_simple(block.get("text", ""))

        if not text or is_page_number(text):
            label = "empty"

        elif is_top_title_candidate(blocks, i) and not seen_first_section:
            label = "document_title"

        else:
            label = classify_line(
                block=block,
                body_font_size=body_font_size,
                blocks=blocks,
                index=i,
            )

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


def build_llm_label_only_prompt(blocks_text: str) -> str:
    return f"""
You are an expert at reading Data Management Plans.

Your task is to classify each CURRENT block using:
- previous_text
- current_text
- next_text
- visual_hint
- rule_label

Return ONLY valid JSON.
Do NOT rewrite text.
Do NOT summarize text.
Do NOT invent headings.
Do NOT omit any block_id.

Allowed labels:
- document_title
- section
- subsection
- content

Definitions:

document_title:
The main title of the DMP. Usually appears at the beginning of page 1.
It may span multiple consecutive beginning blocks.

section:
A major heading that starts a new DMP topic.
It may be numbered, uppercase, bold, larger font, or a short standalone heading followed by paragraph content.

subsection:
A smaller prompt/question/internal heading inside a section.
Often lettered, decimal-numbered, or a short prompt ending with a colon.

content:
Paragraphs, guidance text, answer text, explanation, list items, examples, repository names, and wrapped continuation lines.

Reason like this:

1. Ask: Does current_text start a new idea?
If yes, it may be section or subsection.

2. Ask: Does current_text continue previous_text?
If yes, it is content.

3. If previous_text ends with comma, "and", "or", "of", "to", or "in",
then current_text is usually content.

4. If current_text is short and standalone, and next_text is a long paragraph,
then current_text is probably section.

5. If current_text starts with:
(i), (ii), (iii), -, •
then it is content.

6. If current_text is a data/example line such as:
"Objective sedentary behavior metrics – no existing standards"
then it is content.

7. If current_text is paragraph-like, it is content.

8. Do not classify wrapped sentence fragments as section.

9. Do not classify repository names or institution names alone as subsection.

10. Use rule_label as a hint, not as truth.

Examples:
- "DATA MANAGEMENT AND SHARING PLAN" -> document_title
- "Element 1: Data Type:" -> section
- "1. Data sharing and preservation" -> section
- "Products of Research" followed by a paragraph -> section
- "A. Types and amount of scientific data expected to be generated in the project:" -> subsection
- "(iii) NHANES cohorts" -> content
- "DMPs that explicitly or implicitly commit data management resources..." after "In particular," -> content

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


def get_visual_hint(block: dict, body_font_size: float) -> str:
    text = normalize_text_simple(block.get("text", ""))
    font_size = get_font_size(block)

    if is_bold_block(block):
        return "bold"

    if font_size >= body_font_size + 1 or font_size >= body_font_size * 1.12:
        return "larger_than_body"

    if text.isupper() and len(text.split()) <= 18:
        return "uppercase_heading_like"

    return "body"


def compact_blocks_for_labeling(pdf_blocks: list[dict]) -> list[dict]:
    compact_blocks = []
    body_font_size = get_body_font_size(pdf_blocks)

    clean_pdf_blocks = []

    for block in pdf_blocks:
        text = normalize_text_simple(block.get("text", ""))

        if not text or is_page_number(text):
            continue

        clean_pdf_blocks.append(block)

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
                "rule_label": block.get("label"),
            }
        )

    return compact_blocks


def split_inline_numbered_section(text: str) -> list[dict]:
    text = normalize_text_simple(text)

    pattern = r"^(\d+\.\s+[^.]+\.)\s+(.+)$"
    match = re.match(pattern, text)

    if not match:
        return []

    return [
        {"label": "section", "text": match.group(1).strip()},
        {"label": "content", "text": match.group(2).strip()},
    ]


def split_inline_dot_subsection(text: str) -> list[dict]:
    text = normalize_text_simple(text)

    pattern = r"^([A-Z][A-Za-z0-9/&,\-\(\),\s]{2,90}\.)\s+(.+)$"
    match = re.match(pattern, text)

    if not match:
        return []

    subsection_text = match.group(1).strip()
    content_text = match.group(2).strip()

    heading_words = subsection_text.rstrip(".").split()

    if not (2 <= len(heading_words) <= 10):
        return []

    if len(content_text.split()) < 5:
        return []

    if is_probable_paragraph(subsection_text):
        return []

    return [
        {"label": "subsection", "text": subsection_text},
        {"label": "content", "text": content_text},
    ]


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

        if is_page_number(text) or is_punctuation_only(text):
            continue

        prev_text = clean_output[-1]["text"] if clean_output else ""

        if label == "document_title":
            if document_title_used:
                if clean_output and clean_output[-1]["label"] == "document_title":
                    label = "document_title"
                else:
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

        elif is_decimal_subsection(text):
            label = "subsection"

        elif is_lettered_subsection(text):
            label = "subsection"

        elif label == "subsection" and is_probable_paragraph(text):
            label = "content"

        elif label == "section" and is_false_section_fragment(text, prev_text):
            label = "content"

        if label == "section" and is_inline_prompt_heading(text):
            label = "content"

        if label == "subsection":
            words = text.rstrip(".:").split()

            if len(words) < 3 and not is_lettered_subsection(text):
                label = "content"

            if text and text[0].islower():
                label = "content"

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
    blocks = merge_consecutive_document_title_blocks(blocks)
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
        text = normalize_text_simple(original_block.get("current_text", ""))

        if not text or is_page_number(text) or is_punctuation_only(text):
            continue

        rule_label = normalize_label(original_block.get("rule_label", ""))

        label = label_lookup.get(
            block_id,
            rule_label or "content",
        )

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
    return generate_structured_blocks_with_llm_labels_only(
        llm=llm,
        pdf_blocks=pdf_blocks,
    )


def save_blocks(blocks: list[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)