from pathlib import Path
from typing import List, Dict, Any
import copy

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def make_textarea_answer(answer_id: int, answer_text: str | None = None) -> Dict[str, Any]:
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
    current_answer = question["answer"]["json"]["answer"]

    if current_answer == "Not answered":
        question["answer"]["json"]["answer"] = text
    else:
        question["answer"]["json"]["answer"] = current_answer + "\n" + text


def build_narrative_json_from_blocks(
    structured_blocks: List[Dict],
    skeleton_path: str | Path | None = None
) -> Dict[str, Any]:

    project_root = get_project_root()

    if skeleton_path is None:
        skeleton_path = project_root / "schemas" / "rda_dmp_dmptool_extension_skeleton.json"
    else:
        skeleton_path = Path(skeleton_path)

    skeleton = load_json(skeleton_path)
    output = copy.deepcopy(skeleton)

    sections = []
    current_section = None
    current_question = None

    section_id = 0
    question_id = 0
    answer_id = 0

    for block in structured_blocks:
        label = block.get("label")
        text = block.get("text", "").strip()

        if not text or label == "empty":
            continue

        if label == "document_title":
            output["narrative"]["template"]["title"] = text
            continue

        if label == "section":
            section_id += 1

            current_section = {
                "id": section_id,
                "title": text,
                "description": None,
                "order": section_id,
                "question": []
            }

            sections.append(current_section)
            current_question = None

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

            current_question = {
                "id": question_id,
                "text": text,
                "order": question_order,
                "answer": make_textarea_answer(
                    answer_id=answer_id,
                    answer_text=None
                )
            }

            current_section["question"].append(current_question)

        else:
            if current_question is not None:
                append_text_to_answer(current_question, text)

            elif current_section is not None:
                # Section has answer text but no explicit A/B/C subsection.
                # Create one default question so narrative text goes into answer.json.answer.
                question_id += 1
                answer_id += 1

                current_question = {
                    "id": question_id,
                    "text": current_section["title"],
                    "order": 1,
                    "answer": make_textarea_answer(
                        answer_id=answer_id,
                        answer_text=text
                    )
                }

                current_section["question"].append(current_question)

    output["narrative"]["template"]["section"] = sections

    return output


def save_narrative_json(
    structured_blocks: List[Dict],
    output_path: str | Path,
    skeleton_path: str | Path | None = None
) -> Dict[str, Any]:

    narrative_json = build_narrative_json_from_blocks(
        structured_blocks=structured_blocks,
        skeleton_path=skeleton_path
    )

    output_path = Path(output_path)
    save_json(narrative_json, output_path)

    log(f"Saved narrative JSON: {output_path}")

    return narrative_json