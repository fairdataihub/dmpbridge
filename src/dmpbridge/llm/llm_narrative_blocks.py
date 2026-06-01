# src/dmpbridge/llm/llm_narrative_blocks.py

import json
from pathlib import Path
from json_repair import repair_json


def build_llm_blocks_prompt(dmp_text: str) -> str:
    return f"""
You are helping structure a Data Management Plan.

Convert the DMP text into a JSON list of structured blocks.

Return ONLY valid JSON in this exact format:

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

Allowed labels:
- document_title
- section
- subsection
- content

Rules:
1. Do not invent text.
2. Preserve original wording.
3. Use "document_title" only for the main DMP title.
4. Use "section" for major DMP headings.
5. Use "subsection" for question-like prompts under a section.
6. Use "content" for answer/body text.
7. Do not include page numbers.
8. Do not include markdown or explanation.
9. Return only valid JSON.

DMP text:
{dmp_text}
"""


def extract_json_array(model_output: str) -> str:
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def generate_structured_blocks_with_llm(llm, dmp_text: str):
    prompt = build_llm_blocks_prompt(dmp_text)

    response = llm.invoke(prompt)
    text = response.content

    json_text = extract_json_array(text)

    try:
        blocks = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        blocks = json.loads(repaired)

    clean_blocks = []

    for block in blocks:
        label = block.get("label", "").strip()
        text = block.get("text", "").strip()

        if label and text:
            clean_blocks.append(
                {
                    "label": label,
                    "text": text
                }
            )

    return clean_blocks


def save_blocks(blocks, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)