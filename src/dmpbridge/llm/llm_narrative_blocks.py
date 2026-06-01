# src/dmpbridge/llm/llm_narrative_blocks.py

import json
import re
from pathlib import Path
from json_repair import repair_json


ALLOWED_LABELS = {
    "document_title",
    "section",
    "subsection",
    "content",
}


def build_llm_blocks_prompt(dmp_text: str) -> str:
    return f"""
You are helping structure a Data Management Plan.

Convert the DMP text into a JSON list of structured blocks.

Return ONLY valid JSON.

Allowed labels:
- document_title
- section
- subsection
- content

Definitions:
- document_title: the main title of the DMP.
- section: a major DMP heading.
- subsection: an explicit sub-question or prompt that appears in the original text.
- content: body text, explanation, answer text, or paragraph text.

Rules:
1. Do not invent text.
2. Preserve original wording.
3. Do not summarize text.
4. Do not create new subsection titles.
5. Do not use your own wording.
6. Use "document_title" only for the main DMP title.
7. Use "section" for major numbered headings and Element headings.
8. Use "subsection" only when the exact text appears as a sub-question or prompt.
9. Use "content" for answer/body paragraphs.
10. If a paragraph is a full sentence, label it as "content", not "subsection".
11. Do not include page numbers.
12. Do not include markdown or explanation.
13. Return only valid JSON.

Important examples from the target PDFs:

Example A: NIH Element format
Input:
DATA MANAGEMENT AND SHARING PLAN
Element 1: Data Type:
A. Types and amount of scientific data expected to be generated in the project:
This secondary data analysis project will analyze deidentified data...
Element 2: Related Tools, Software and/or Code:
Data will be analyzed with custom code...

Expected labels:
[
  {{
    "label": "document_title",
    "text": "DATA MANAGEMENT AND SHARING PLAN"
  }},
  {{
    "label": "section",
    "text": "Element 1: Data Type:"
  }},
  {{
    "label": "subsection",
    "text": "A. Types and amount of scientific data expected to be generated in the project:"
  }},
  {{
    "label": "content",
    "text": "This secondary data analysis project will analyze deidentified data..."
  }},
  {{
    "label": "section",
    "text": "Element 2: Related Tools, Software and/or Code:"
  }},
  {{
    "label": "content",
    "text": "Data will be analyzed with custom code..."
  }}
]

Example B: NSF numbered section format
Input:
Univ. of California, Riverside Data Management Plan
1. Policy and Practice
The Bourns College of Engineering...
2. Scope
This Data Management Plan addresses the NSF policy...

Expected labels:
[
  {{
    "label": "document_title",
    "text": "Univ. of California, Riverside Data Management Plan"
  }},
  {{
    "label": "section",
    "text": "1. Policy and Practice"
  }},
  {{
    "label": "content",
    "text": "The Bourns College of Engineering..."
  }},
  {{
    "label": "section",
    "text": "2. Scope"
  }},
  {{
    "label": "content",
    "text": "This Data Management Plan addresses the NSF policy..."
  }}
]

Example C: inline numbered heading format
Input:
Data Management Plan:
1. Types of data. The bulk of the data generated in this project will be 1, 2, 3, and 4 dimensional arrays...
2. Data and metadata standards. The PI’s research group will adopt...

Expected labels:
[
  {{
    "label": "document_title",
    "text": "Data Management Plan:"
  }},
  {{
    "label": "section",
    "text": "1. Types of data."
  }},
  {{
    "label": "content",
    "text": "The bulk of the data generated in this project will be 1, 2, 3, and 4 dimensional arrays..."
  }},
  {{
    "label": "section",
    "text": "2. Data and metadata standards."
  }},
  {{
    "label": "content",
    "text": "The PI’s research group will adopt..."
  }}
]

Very important:
- For PDFs like Sample 8 and Sample 10, do NOT create subsections such as "Data Management System", "Data Backup", "Data Exclusions", or "Data Protection" unless those exact headings appear in the source text.
- If text is a paragraph, label it as "content".
- A sentence like "All principal investigators in BCOE are responsible..." is content, not subsection.
- A sentence like "Data that must be withheld long enough..." is content, not subsection.
- A sentence like "Confidential material will be handled..." is content, not subsection.

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

DMP text:
{dmp_text}
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
    """
    Handles Sample 6 style:

    1. Types of data. The bulk of the data generated...
    ->
    section: 1. Types of data.
    content: The bulk of the data generated...
    """
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
    )

    return text.startswith(starters)


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

        # Sample 6 safety:
        # If the LLM keeps "1. Types of data. The bulk..." together,
        # split it into section + content.
        if label in {"section", "content"}:
            split_blocks = split_inline_numbered_section(text)

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

        clean_blocks.append(
            {
                "label": label,
                "text": text,
            }
        )

    return clean_blocks


def generate_structured_blocks_with_llm(llm, dmp_text: str):
    prompt = build_llm_blocks_prompt(dmp_text)

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