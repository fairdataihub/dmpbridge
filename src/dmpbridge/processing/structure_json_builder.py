from pathlib import Path
from typing import List, Dict, Any

from dmpbridge.utils.file_io import save_json
from dmpbridge.utils.logger import log


def build_structure_json(structured_blocks: List[Dict]) -> Dict[str, Any]:
    """
    Convert labeled line blocks into grouped structure:
    section → subsection → content
    """

    result = {
        "source_pdf": structured_blocks[0].get("source_pdf") if structured_blocks else "",
        "structure_type": "section_subsection_content",
        "sections": []
    }

    current_section = None
    current_subsection = None

    for block in structured_blocks:
        label = block.get("label")
        text = block.get("text", "").strip()

        if not text or label == "empty":
            continue

        if label == "section":
            current_section = {
                "order": len(result["sections"]) + 1,
                "title": text,
                "page": block.get("page"),
                "subsections": [],
                "content": []
            }
            result["sections"].append(current_section)
            current_subsection = None

        elif label == "subsection":
            if current_section is None:
                current_section = {
                    "order": len(result["sections"]) + 1,
                    "title": "Untitled Section",
                    "page": block.get("page"),
                    "subsections": [],
                    "content": []
                }
                result["sections"].append(current_section)

            current_subsection = {
                "order": len(current_section["subsections"]) + 1,
                "title": text,
                "page": block.get("page"),
                "content": []
            }
            current_section["subsections"].append(current_subsection)

        else:
            content_item = {
                "text": text,
                "page": block.get("page"),
                "line_order": block.get("line_order"),
                "label": label
            }

            if current_subsection is not None:
                current_subsection["content"].append(content_item)
            elif current_section is not None:
                current_section["content"].append(content_item)

    return result


def save_structure_json(structured_blocks: List[Dict], output_path: str | Path) -> Dict[str, Any]:
    structure = build_structure_json(structured_blocks)

    output_path = Path(output_path)
    save_json(structure, output_path)

    log(f"Saved structure JSON: {output_path}")

    return structure