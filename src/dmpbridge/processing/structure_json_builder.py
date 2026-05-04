from pathlib import Path
from typing import List, Dict, Any
import copy

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_narrative_json_from_blocks(
    structured_blocks: List[Dict],
    skeleton_path: str | Path | None = None
) -> Dict[str, Any]:
    """
    Build full RDA + DMPTool extension JSON skeleton,
    but only fill narrative.template.section.
    """

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

    for block in structured_blocks:
        label = block.get("label")
        text = block.get("text", "").strip()

        if not text or label == "empty":
            continue

        if label == "section":
            current_section = {
                "id": f"section_{len(sections) + 1}",
                "title": text,
                "description": None,
                "order": len(sections) + 1,
                "question": []
            }

            sections.append(current_section)
            current_question = None

        elif label == "subsection":
            if current_section is None:
                current_section = {
                    "id": f"section_{len(sections) + 1}",
                    "title": "Untitled Section",
                    "description": None,
                    "order": len(sections) + 1,
                    "question": []
                }
                sections.append(current_section)

            current_question = {
                "id": f"question_{current_section['order']}_{len(current_section['question']) + 1}",
                "text": text,
                "order": len(current_section["question"]) + 1,
                "answer": {
                    "id": f"answer_{current_section['order']}_{len(current_section['question']) + 1}",
                    "json": {
                        "type": "text",
                        "answer": [
                            {
                                "text": ""
                            }
                        ],
                        "meta": {
                            "schemaVersion": None
                        }
                    }
                }
            }

            current_section["question"].append(current_question)

        else:
            # content/question/instruction lines become answer text
            if current_question is not None:
                answer_list = current_question["answer"]["json"]["answer"]
                existing_text = answer_list[0].get("text", "")

                if existing_text:
                    answer_list[0]["text"] = existing_text + "\n" + text
                else:
                    answer_list[0]["text"] = text

            elif current_section is not None:
                # content before first subsection becomes section description
                existing_description = current_section.get("description")

                if existing_description:
                    current_section["description"] = existing_description + "\n" + text
                else:
                    current_section["description"] = text

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