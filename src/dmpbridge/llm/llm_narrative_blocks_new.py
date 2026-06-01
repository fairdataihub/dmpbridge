# src/dmpbridge/llm/llm_narrative_blocks.py

import json
import re
from pathlib import Path
from json_repair import repair_json
from dmpbridge.processing.text_cleaner import clean_repeated_words


ALLOWED_LABELS = {
    "document_title",
    "section",
    "subsection",
    "content",
}


def build_llm_blocks_prompt(blocks_text: str) -> str:
    return f"""
You are helping extract existing narrative blocks from a Data Management Plan.

Your task is BLOCK EXTRACTION ONLY.



Allowed labels:
- document_title
- section
- subsection
- content

Definitions:
- document_title: the main title of the DMP.
- section: an existing major heading copied exactly from the input block.
- subsection: an existing prompt/question/subheading copied exactly from the input block.
- content: body text, instruction text, guidance text, answer text, explanation, or paragraph text.

You are given PDFPlumber extracted blocks. Each block may include:
- block_id
- text
- page
- line_order
- x0, top, x1, bottom
- avg_font_size
- font_names
- is_bold

Use ALL available layout clues:
- Larger font size often indicates document_title or section.
- Bold text may indicate title, section, or subsection.
- Mixed font_names may mean the line contains a bold heading followed by normal content.
- Lower x0 or indentation may indicate list items or body content.
- page and line_order show reading order.
- Long full-sentence blocks are usually content, even if bold.

Critical extraction rules:
1. Do not invent text.
2. Do not copy headings from examples.
3. Do not infer hidden headings.
4. Do not create new headings.
5. Do not summarize content into headings.
6. Do not rename headings.
7. Preserve original wording and punctuation.
8. A section or subsection must appear explicitly in the input block text.
9. If unsure, label as content.
10. Do not include page numbers.
11. Return only a valid JSON array.

How to detect document_title:
- Usually appears near the beginning.
- Often bold.
- Often larger than body text.
- Usually short and describes the whole DMP.

How to detect section:
Use section for existing major DMP headings:
- numbered headings, e.g., "1. Data sharing and preservation"
- element headings, e.g., "Element 1: Data Type:"
- short standalone major category headings
- bold/larger-font heading lines

A section must be a heading, not a normal sentence.

How to detect subsection:
Use subsection for existing prompts inside a section:
- lettered prompts, e.g., "A. Types and amount..."
- short dot-ended or colon-ended prompts inside a section
- bold phrase at the start of a mixed-font line

If a block starts with a short heading followed by content, split it:
Example:
"Roles & Responsibilities. For the proposed research..."
becomes:
[
  {{
    "label": "subsection",
    "text": "Roles & Responsibilities."
  }},
  {{
    "label": "content",
    "text": "For the proposed research..."
  }}
]

How to detect content:
Use content for:
- full sentences
- paragraphs
- guidance/instruction text
- answer text
- body text after a heading
- body text after a prompt
- bold guidance paragraphs that are not headings
- list items such as "(i) The RISE study"

Important negative rules:
- Do not turn normal sentences into sections.
- Do not turn paragraphs into subsections.
- Do not create headings like "Data Backup", "Data Protection", "Data Exclusions", or "Data Management System" unless that exact heading appears in the input.
- Text starting with "Data management plans should...", "The Data Management Plan should...", "DMPs should...", "Data will...", "We will...", or "The proposed..." is usually content.

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

def extract_json_array(model_output: str) -> str:
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", clean_repeated_words(text)))


def is_element_section(text: str) -> bool:
    return bool(
        re.match(
            r"^Element\s+\d+\s*:",
            clean_repeated_words(text),
            flags=re.IGNORECASE,
        )
    )


def is_numbered_section(text: str) -> bool:
    return bool(
        re.match(
            r"^\d+\.\s+[A-Z]",
            clean_repeated_words(text),
        )
    )


def is_lettered_subsection(text: str) -> bool:
    return bool(
        re.match(
            r"^[A-Z]\.\s+.+",
            clean_repeated_words(text),
        )
    )


def is_list_item(text: str) -> bool:
    text = str(text).strip()

    return bool(
        re.match(
            r"^\((?:[ivxlcdm]+|\d+|[a-z])\)\s+.+",
            text,
            flags=re.IGNORECASE,
        )
    )


def split_inline_numbered_section(text: str) -> list[dict]:
    text = clean_repeated_words(text)
    match = re.match(r"^(\d+\.\s+[^.]+\.)\s+(.+)$", text)

    if not match:
        return []

    return [
        {"label": "section", "text": match.group(1).strip()},
        {"label": "content", "text": match.group(2).strip()},
    ]


def split_inline_dot_subsection(text: str) -> list[dict]:
    text = clean_repeated_words(text)
    match = re.match(r"^([A-Z][A-Za-z/&,\-\s]+?\.)\s+(.+)$", text)

    if not match:
        return []

    subsection_text = match.group(1).strip()
    content_text = match.group(2).strip()

    if len(subsection_text.split()) > 8:
        return []

    return [
        {"label": "subsection", "text": subsection_text},
        {"label": "content", "text": content_text},
    ]


def is_sentence_or_paragraph(text: str) -> bool:
    text = clean_repeated_words(text)
    words = text.split()

    if len(words) > 12:
        return True

    starters = (
        "The ", "This ", "These ", "All ", "We ", "Data ",
        "Materials ", "Interested ", "Local ", "Stored ",
        "If ", "As ", "In ", "Additionally,", "Researchers ",
        "Confidential ", "Software ", "Select ", "Research ",
        "Upon ", "Our ", "First,", "Second,", "Specifically,",
    )

    return text.startswith(starters)


def compact_pdfplumber_blocks(pdf_blocks):
    compact_blocks = []

    for i, block in enumerate(pdf_blocks, start=1):
        text = clean_repeated_words(str(block.get("text", ""))).strip()

        if not text:
            continue

        compact_blocks.append(
            {
                "block_id": i,
                "text": text,
                "page": block.get("page"),
                "line_order": block.get("line_order"),
                "x0": block.get("x0"),
                "top": block.get("top"),
                "x1": block.get("x1"),
                "bottom": block.get("bottom"),
                "avg_font_size": block.get("avg_font_size"),
                "font_names": block.get("font_names"),
                "is_bold": block.get("is_bold"),
            }
        )

    return compact_blocks

def looks_like_continuation_fragment(text: str) -> bool:
    text = clean_repeated_words(str(text)).strip()

    if not text:
        return False

    if is_element_section(text):
        return False

    if is_numbered_section(text):
        return False

    if is_lettered_subsection(text):
        return False

    words = text.split()

    return (
        len(words) <= 8
        and text.endswith(".")
        and not text.endswith(":")
    )
    
def postprocess_blocks(blocks):
    clean_blocks = []
    document_title_used = False

    for block in blocks:
        if not isinstance(block, dict):
            continue

        label = normalize_label(block.get("label", ""))
        text = clean_repeated_words(str(block.get("text", ""))).strip()

        if not label or not text:
            continue

        if is_page_number(text):
            continue

        if label == "document_title":
            if document_title_used:
                label = "content"
            else:
                document_title_used = True

        if label in {"section", "content"}:
            split_blocks = split_inline_numbered_section(text)
            if split_blocks:
                clean_blocks.extend(split_blocks)
                continue

        if label in {"subsection", "content"}:
            split_blocks = split_inline_dot_subsection(text)
            if split_blocks:
                clean_blocks.extend(split_blocks)
                continue

        if is_element_section(text):
            label = "section"
        elif is_numbered_section(text):
            label = "section"
        elif is_lettered_subsection(text):
            label = "subsection"
        elif label == "subsection" and is_sentence_or_paragraph(text):
            label = "content"

        if label == "section" and is_sentence_or_paragraph(text):
            label = "content"
            
        if looks_like_continuation_fragment(text):
            label = "content"

        if is_list_item(text):
            label = "content"

        clean_blocks.append(
            {
                "label": label,
                "text": text,
            }
        )

    return clean_blocks


def generate_structured_blocks_with_llm(llm, pdf_blocks):
    compact_blocks = compact_pdfplumber_blocks(pdf_blocks)

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

    return postprocess_blocks(blocks)


def save_blocks(blocks, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)