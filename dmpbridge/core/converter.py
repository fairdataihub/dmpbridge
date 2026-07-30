"""Convert flat labeled blocks (LLM output) to the DMP Tool narrative JSON schema."""

# What this file does — step by step:
#   Step 1 — walk every labeled block in document reading order
#   Step 2 — section.title opens a new section in the output
#   Step 3 — section.description before any question → goes into the section description field
#   Step 4 — section.description after a question has started → stays in sequence (continues the question or answer)
#   Step 5 — consecutive question.text blocks with no answer yet → merged into one question
#   Step 6 — answer.text → appended to the current question's answer
#   Step 7 — post-process: every section must have at least one question (Rules 1–3 below)
#   Step 8 — return the fully nested JSON: narrative → template → section[] → question[] → answer
#
# Question.text fallback rules (PM requirement — question.text must never be empty):
#   Rule 1: section with no questions → create one question using the section title as text
#   Rule 2: section with empty title and no questions → use document title as fallback
#   Rule 3: document with no sections at all → create one section+question from the document title

import json
from pathlib import Path
from typing import Union


def to_structured(blocks: list[dict], pdf_url: str = "") -> dict:
    """Convert a flat list of labeled blocks into the nested DMP Tool JSON schema."""

    title = ""
    sections: list[dict] = []
    cur_section: dict | None = None
    cur_question: dict | None = None
    sec_order = 0
    q_order = 0

    def _new_section(sec_title: str) -> dict:
        # Create a fresh section dict and reset the question counter.
        nonlocal sec_order, q_order
        sec_order += 1
        q_order = 0
        return {
            "title": sec_title,
            "description": "",
            "order": sec_order,
            "question": [],
        }

    def _new_question(q_text: str) -> dict:
        # Create a fresh question dict with an empty answer ready to be filled.
        nonlocal q_order
        q_order += 1
        return {
            "text": q_text,
            "order": q_order,
            "answer": {
                "json": {
                    "type": "textArea",
                    "answer": "",
                    "meta": {
                        "schemaVersion": "1.0",
                    },
                }
            },
        }

    for block in blocks:
        label = block.get("label", "answer.text")
        text = block.get("text", "").strip()
        if not text:
            continue

        if label == "title":
            if not title:
                # First title block becomes the document title.
                title = text
            elif not sections:
                # Another title before any section — pdfplumber split the title across lines, merge it.
                title = title + " " + text
            else:
                # Title appearing after sections have started — treat as answer content.
                if cur_question is None:
                    if cur_section is None:
                        cur_section = _new_section("")
                        sections.append(cur_section)
                    cur_question = _new_question("")
                    cur_section["question"].append(cur_question)
                existing = cur_question["answer"]["json"]["answer"]
                cur_question["answer"]["json"]["answer"] = (
                    existing + "\n" + text if existing else text
                )

        elif label == "section.title":
            # Start a new section and reset the current question pointer.
            cur_section = _new_section(text)
            sections.append(cur_section)
            cur_question = None

        elif label == "section.description":
            if cur_section is None:
                cur_section = _new_section("")
                sections.append(cur_section)
            if cur_question is None:
                # No question open yet — this is a true leading description for the section.
                if cur_section["description"]:
                    cur_section["description"] += "\n" + text
                else:
                    cur_section["description"] = text
            elif not cur_question["answer"]["json"]["answer"]:
                # A question is open but has no answer yet — stay in sequence, continue its text.
                cur_question["text"] = (cur_question["text"] + "\n" + text) if cur_question["text"] else text
            else:
                # A question already has an answer — stay in sequence, continue the answer.
                cur_question["answer"]["json"]["answer"] += "\n" + text

        elif label == "question.text":
            if cur_section is None:
                cur_section = _new_section("")
                sections.append(cur_section)
            if cur_question is not None and not cur_question["answer"]["json"]["answer"]:
                # Previous block was also a question with no answer yet — merge into one question.
                cur_question["text"] = (cur_question["text"] + "\n" + text) if cur_question["text"] else text
            else:
                # New question — create a fresh entry under the current section.
                cur_question = _new_question(text)
                cur_section["question"].append(cur_question)

        elif label == "answer.text":
            if cur_question is None:
                # Answer appeared before any question — create an implicit empty question to hold it.
                if cur_section is None:
                    cur_section = _new_section("")
                    sections.append(cur_section)
                cur_question = _new_question("")
                cur_section["question"].append(cur_question)
            # Append to the current question's answer, joining multiple answer blocks with a newline.
            existing = cur_question["answer"]["json"]["answer"]
            cur_question["answer"]["json"]["answer"] = (
                existing + "\n" + text if existing else text
            )

    # ── Post-process: ensure every section has at least one question ─────────────
    # Rule 1/2: section exists but has no questions → use section title as question.text.
    # Rule 3: no sections at all → create one section+question from the document title.
    if not sections and title:
        s = {
            "title": title,
            "description": "",
            "order": 1,
            "question": [
                {
                    "text": title,
                    "order": 1,
                    "answer": {
                        "json": {
                            "type": "textArea",
                            "answer": "",
                            "meta": {"schemaVersion": "1.0"},
                        }
                    },
                }
            ],
        }
        sections.append(s)
    else:
        for sec in sections:
            fallback = sec["title"] or title
            if not sec["question"]:
                # No questions at all → create one from the section title.
                sec["question"].append(
                    {
                        "text": fallback,
                        "order": 1,
                        "answer": {
                            "json": {
                                "type": "textArea",
                                "answer": "",
                                "meta": {"schemaVersion": "1.0"},
                            }
                        },
                    }
                )
            else:
                # Questions exist but some may have empty text (implicit placeholders
                # created when answer.text appeared before any question.text block).
                for q in sec["question"]:
                    if not q["text"]:
                        q["text"] = fallback

    return {
        "narrative": {
            "download_url": pdf_url,
            "template": {
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
    pdf_url: str = "",
) -> dict:
    """Load a flat labeled JSON file and write the structured JSON to disk."""
    flat_path = Path(flat_path)
    # Read the flat labeled JSON produced by the LLM classifier.
    blocks = json.loads(flat_path.read_text(encoding="utf-8"))
    structured = to_structured(blocks, pdf_url=pdf_url)

    # Default output path is the same folder as the input, with _structured appended.
    if structured_path is None:
        structured_path = flat_path.with_name(flat_path.stem + "_structured.json")
    structured_path = Path(structured_path)
    structured_path.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return structured
