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
You are helping extract the existing narrative structure from a Data Management Plan.

Your task is extraction only.

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
1. Do not invent text.
2. Do not infer hidden headings.
3. Do not create new headings.
4. Do not summarize text into headings.
5. Do not rename headings.
6. Do not use your own wording.
7. Preserve original wording as much as possible.
8. A section or subsection must appear explicitly in the input blocks.
9. If unsure, label the text as "content".
10. Do not include page numbers.
11. Do not include markdown or explanation.
12. Return only the block array.

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

A section must be a heading, not a normal sentence.

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

Do NOT label normal sentences as sections or subsections.

Important negative rules:
- Do not create headings like "Data Backup", "Data Protection", "Data Exclusions", "Data Management System", or "Software generated under the project" unless that exact heading appears as a heading in the input blocks.
- Do not turn a sentence into a heading just because it introduces a topic.
- Do not turn "Data will...", "We will...", "Software generated...", or "The proposed..." sentences into sections.
- A paragraph should never become a section.
- A paragraph should never become a subsection.

Inline heading rule:
If a real heading and its content appear in the same block, split them into two blocks:
1. section or subsection
2. content

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


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", text.strip()))


def is_element_section(text: str) -> bool:
    return bool(
        re.match(
            r"^Element\s+\d+\s*:",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def is_numbered_section(text: str) -> bool:
    return bool(
        re.match(
            r"^\d+\.\s+[A-Z]",
            text.strip(),
        )
    )


def is_lettered_subsection(text: str) -> bool:
    return bool(
        re.match(
            r"^[A-Z]\.\s+.+",
            text.strip(),
        )
    )


def split_inline_numbered_section(text: str) -> list[dict]:
    text = text.strip()

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
    text = text.strip()

    pattern = r"^([A-Z][A-Za-z/&,\-\s]+?\.)\s+(.+)$"
    match = re.match(pattern, text)

    if not match:
        return []

    subsection_text = match.group(1).strip()
    content_text = match.group(2).strip()

    if len(subsection_text.split()) > 8:
        return []

    if not subsection_text or not content_text:
        return []

    return [
        {
            "label": "subsection",
            "text": subsection_text,
        },
        {
            "label": "content",
            "text": content_text,
        },
    ]


def is_sentence_or_paragraph(text: str) -> bool:
    text = text.strip()
    words = text.split()

    if len(words) > 12:
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


def compact_pdfplumber_blocks(pdf_blocks):
    compact_blocks = []

    for i, block in enumerate(pdf_blocks, start=1):
        text = str(block.get("text", "")).strip()

        if not text:
            continue

        compact_blocks.append(
            {
                "block_id": i,
                "text": text,
                "page": block.get("page"),
                "font_size": block.get("font_size"),
                "is_bold": block.get("is_bold"),
            }
        )

    return compact_blocks


def postprocess_blocks(blocks):
    clean_blocks = []
    document_title_used = False

    for block in blocks:
        if not isinstance(block, dict):
            continue

        label = normalize_label(block.get("label", ""))
        text = str(block.get("text", "")).strip()

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