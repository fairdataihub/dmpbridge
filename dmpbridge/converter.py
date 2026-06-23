"""Convert flat labeled blocks (LLM output) to the hierarchical manual annotation schema."""

import json
from pathlib import Path
from typing import Union


def to_structured(blocks: list[dict]) -> dict:
    """
    Convert a flat list of labeled blocks into the same hierarchical JSON schema
    used by the manual annotations in data/manuallabeled/.

    Schema
    ------
    narrative.template
      title
      section[]
        title
        description        (section.description blocks joined with \\n)
        order
        question[]
          text             (question.text block)
          order
          answer.json.answer  (answer.text blocks joined with space)

    Edge cases
    ----------
    - answer.text before any question.text  → implicit question with empty text
    - question.text before any section.title → implicit section with empty title
    - Multiple consecutive section.description blocks → joined with \\n
    - Multiple consecutive answer.text blocks → joined with a single space
      (pdfplumber splits one paragraph into many short lines)
    """
    title = ""
    sections: list[dict] = []
    cur_section: dict | None = None
    cur_question: dict | None = None
    sec_order = 0
    q_order = 0

    def _new_section(sec_title: str) -> dict:
        nonlocal sec_order, q_order
        sec_order += 1
        q_order = 0
        return {
            "id": "",
            "title": sec_title,
            "description": "",
            "order": sec_order,
            "question": [],
        }

    def _new_question(q_text: str) -> dict:
        nonlocal q_order
        q_order += 1
        return {
            "id": "",
            "text": q_text,
            "order": q_order,
            "answer": {
                "id": "",
                "json": {
                    "type": "textArea",
                    "answer": "",
                    "meta": {"schemaVersion": "1.0"},
                },
            },
        }

    for block in blocks:
        label = block.get("label", "answer.text")
        text = block.get("text", "").strip()
        if not text:
            continue

        if label == "title":
            title = text

        elif label == "section.title":
            cur_section = _new_section(text)
            sections.append(cur_section)
            cur_question = None

        elif label == "section.description":
            if cur_section is None:
                cur_section = _new_section("")
                sections.append(cur_section)
            if cur_section["description"]:
                cur_section["description"] += "\n" + text
            else:
                cur_section["description"] = text

        elif label == "question.text":
            if cur_section is None:
                cur_section = _new_section("")
                sections.append(cur_section)
            cur_question = _new_question(text)
            cur_section["question"].append(cur_question)

        elif label == "answer.text":
            if cur_question is None:
                if cur_section is None:
                    cur_section = _new_section("")
                    sections.append(cur_section)
                cur_question = _new_question("")
                cur_section["question"].append(cur_question)
            existing = cur_question["answer"]["json"]["answer"]
            cur_question["answer"]["json"]["answer"] = (
                existing + " " + text if existing else text
            )

    return {
        "narrative": {
            "download_url": "",
            "template": {
                "id": "",
                "title": title,
                "description": "",
                "version": "v1",
                "section": sections,
            },
        }
    }


def convert_file(
    flat_path: Union[str, Path],
    structured_path: Union[str, Path, None] = None,
) -> dict:
    """
    Load a flat labeled JSON file and return (and optionally save) the structured version.

    Parameters
    ----------
    flat_path       : path to *_<model>.json produced by the pipeline
    structured_path : if given, write the structured JSON here;
                      defaults to <stem>_structured.json next to the input
    """
    flat_path = Path(flat_path)
    blocks = json.loads(flat_path.read_text(encoding="utf-8"))
    structured = to_structured(blocks)

    if structured_path is None:
        structured_path = flat_path.with_name(flat_path.stem + "_structured.json")
    structured_path = Path(structured_path)
    structured_path.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return structured
