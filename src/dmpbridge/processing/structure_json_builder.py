from pathlib import Path
from typing import List, Dict, Any
import copy
import re

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    """
    Return the root folder of the dmpbridge project.

    Current file:
    src/dmpbridge/processing/structure_json_builder.py
    """
    return Path(__file__).resolve().parents[3]


def make_textarea_answer(answer_id: int, answer_text: str | None = None) -> Dict[str, Any]:
    """
    Create a DMPTool-compatible answer object.

    Default answer type is textArea because most extracted DMP narrative
    answers are free text.
    """
    return {
        "id": answer_id,
        "json": {
            "type": "textArea",
            "meta": {
                "schemaVersion": "1.0"
            },
            "answer": answer_text.strip() if answer_text and answer_text.strip() else "Not answered"
        }
    }


def append_text_to_answer(question: Dict[str, Any], text: str) -> None:
    """
    Append content text to the current question answer.
    """
    current_answer = question["answer"]["json"]["answer"]

    if current_answer == "Not answered":
        question["answer"]["json"]["answer"] = text
    else:
        question["answer"]["json"]["answer"] = current_answer + "\n" + text


def split_numbered_heading_and_answer(text: str) -> tuple[str, str | None]:
    """
    Split inline section headings that contain answer text on the same line.

    Examples:
    1. Types of data. The bulk of the data...
    -> title: 1. Types of data
    -> answer: The bulk of the data...

    4. Use of Vertebrate Animals: Vertebrate animals...
    -> title: 4. Use of Vertebrate Animals
    -> answer: Vertebrate animals...
    """

    # Colon-style inline heading
    colon_pattern = r"^(\d+\.\s+[A-Z][^:]+):\s+(.*)$"
    colon_match = re.match(colon_pattern, text)

    if colon_match:
        title = colon_match.group(1).strip()
        answer = colon_match.group(2).strip()
        return title, answer

    # Period-style inline heading
    period_pattern = r"^(\d+\.\s+[A-Z][^.]+)\.\s+(.*)$"
    period_match = re.match(period_pattern, text)

    if period_match:
        title = period_match.group(1).strip()
        answer = period_match.group(2).strip()
        return title, answer

    return text, None


def is_page_number(block: Dict[str, Any], text: str) -> bool:
    """
    Skip standalone page numbers such as 1, 2, 3.
    """
    return text.isdigit() and block.get("page") is not None


def create_default_question(
    question_id: int,
    answer_id: int,
    question_text: str,
    answer_text: str | None = None,
    question_order: int = 1
) -> Dict[str, Any]:
    """
    Create one question object with one textArea answer.
    """
    return {
        "id": question_id,
        "text": question_text,
        "order": question_order,
        "answer": make_textarea_answer(
            answer_id=answer_id,
            answer_text=answer_text
        )
    }


def create_default_section(
    section_id: int,
    section_title: str,
    question_id: int,
    answer_id: int,
    answer_text: str,
) -> Dict[str, Any]:
    """
    Create a default section with one question and one answer.

    Used when a DMP has no clear section headings, such as a single-paragraph
    Resource/Data Sharing Plan.
    """
    return {
        "id": section_id,
        "title": section_title,
        "description": None,
        "order": section_id,
        "question": [
            create_default_question(
                question_id=question_id,
                answer_id=answer_id,
                question_text=section_title,
                answer_text=answer_text,
                question_order=1
            )
        ]
    }


def build_narrative_json_from_blocks(
    structured_blocks: List[Dict],
    skeleton_path: str | Path | None = None
) -> Dict[str, Any]:
    """
    Build the final RDA + DMPTool narrative JSON from structured blocks.

    Input:
    - structured_blocks from structure_detector.py
    - each block should have a label:
      document_title, section, subsection, question, content, or empty

    Output:
    - full JSON based on rda_dmp_dmptool_extension_skeleton.json
    - fills narrative.template.title
    - fills narrative.template.section[]
    """

    project_root = get_project_root()

    if skeleton_path is None:
        skeleton_path = project_root / "schemas" / "rda_dmp_dmptool_extension_skeleton.json"
    else:
        skeleton_path = Path(skeleton_path)

    # Load the base skeleton and copy it so the original is not modified.
    skeleton = load_json(skeleton_path)
    output = copy.deepcopy(skeleton)

    sections = []
    current_section = None
    current_question = None

    # Incrementing IDs required by the DMPTool narrative extension.
    section_id = 0
    question_id = 0
    answer_id = 0

    for block in structured_blocks:
        label = block.get("label")
        text = block.get("text", "").strip()

        if not text or label == "empty":
            continue

        # Remove page-number-only lines.
        if is_page_number(block, text):
            continue

        # Document title becomes template title, not a section.
        if label == "document_title":
            output["narrative"]["template"]["title"] = text
            continue

        # Start a new section.
        if label == "section":
            section_id += 1

            # Some PDFs put section title and answer on the same line.
            section_title, first_answer = split_numbered_heading_and_answer(text)

            current_section = {
                "id": section_id,
                "title": section_title,
                "description": None,
                "order": section_id,
                "question": []
            }

            sections.append(current_section)
            current_question = None

            # If the section line also included answer text, create a default question.
            if first_answer:
                question_id += 1
                answer_id += 1

                current_question = create_default_question(
                    question_id=question_id,
                    answer_id=answer_id,
                    question_text=section_title,
                    answer_text=first_answer,
                    question_order=1
                )

                current_section["question"].append(current_question)

        # Start a new question inside the current section.
        elif label == "subsection":
            if current_section is None:
                section_id += 1
                current_section = {
                    "id": section_id,
                    "title": "Untitled Section",
                    "description": None,
                    "order": section_id,
                    "question": []
                }
                sections.append(current_section)

            question_id += 1
            answer_id += 1
            question_order = len(current_section["question"]) + 1

            current_question = create_default_question(
                question_id=question_id,
                answer_id=answer_id,
                question_text=text,
                answer_text=None,
                question_order=question_order
            )

            current_section["question"].append(current_question)

        # Content lines become answer text.
        else:
            if current_question is not None:
                append_text_to_answer(current_question, text)

            elif current_section is not None:
                # Section has content but no explicit subsection/question.
                question_id += 1
                answer_id += 1

                current_question = create_default_question(
                    question_id=question_id,
                    answer_id=answer_id,
                    question_text=current_section["title"],
                    answer_text=text,
                    question_order=1
                )

                current_section["question"].append(current_question)

            else:
                # Content appears before any section.
                # This handles one-paragraph DMPs like sample7.
                section_id += 1
                question_id += 1
                answer_id += 1

                section_title = (
                    output["narrative"]["template"].get("title")
                    or "Narrative"
                )

                current_section = create_default_section(
                    section_id=section_id,
                    section_title=section_title,
                    question_id=question_id,
                    answer_id=answer_id,
                    answer_text=text
                )

                current_question = current_section["question"][0]
                sections.append(current_section)

    output["narrative"]["template"]["section"] = sections

    return output


def save_narrative_json(
    structured_blocks: List[Dict],
    output_path: str | Path,
    skeleton_path: str | Path | None = None
) -> Dict[str, Any]:
    """
    Build and save the narrative JSON output.
    """

    narrative_json = build_narrative_json_from_blocks(
        structured_blocks=structured_blocks,
        skeleton_path=skeleton_path
    )

    output_path = Path(output_path)
    save_json(narrative_json, output_path)

    log(f"Saved narrative JSON: {output_path}")

    return narrative_json