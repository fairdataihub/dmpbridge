from pathlib import Path
from typing import List, Dict, Any
import re

from dmpbridge.utils.file_io import save_json
from dmpbridge.utils.logger import log


def clean_text(text: str) -> str:
    return text.strip().replace("&amp;", "&")


def is_generic_document_title(text: str) -> bool:
    return text.strip().lower() in {
        "data management plan",
        "data management and sharing plan",
        "dmp",
    }


def is_nih_element_heading(text: str) -> bool:
    return re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE) is not None


def is_lettered_question(text: str) -> bool:
    return re.match(r"^[A-Z]\.\s+", text) is not None


def markdown_to_structured_blocks(
    markdown_text: str,
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:

    structured_blocks = []
    line_order = 1
    seen_title = False
    seen_real_section = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        label = "content"

        # Remove markdown heading marks
        if line.startswith("#"):
            heading_level = len(line) - len(line.lstrip("#"))
            text = clean_text(line.lstrip("#").strip())
        else:
            heading_level = 0
            text = clean_text(line)

        if not text:
            continue

        # Skip repeated generic title after real sections have started
        if is_generic_document_title(text) and seen_real_section:
            continue

        # First generic DMP title can be template title
        if is_generic_document_title(text) and not seen_title:
            label = "document_title"
            seen_title = True

        elif is_nih_element_heading(text):
            label = "section"
            seen_real_section = True

        elif is_lettered_question(text):
            label = "question"

        elif heading_level == 1 and not seen_title:
            label = "document_title"
            seen_title = True

        elif heading_level in [1, 2]:
            label = "section"
            seen_real_section = True

        elif heading_level >= 3:
            label = "question"

        else:
            label = "content"

        structured_blocks.append({
            "source_pdf": source_pdf,
            "page": None,
            "line_order": line_order,
            "text": text,
            "label": label,
            "document_format": "docling_markdown",
            "extractor": "docling"
        })

        line_order += 1

    return structured_blocks


def save_docling_structured_blocks(
    markdown_path: str | Path,
    output_path: str | Path,
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:

    markdown_path = Path(markdown_path)
    output_path = Path(output_path)

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    structured_blocks = markdown_to_structured_blocks(
        markdown_text=markdown_text,
        source_pdf=source_pdf
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(structured_blocks, output_path)

    log(f"Saved Docling structured blocks: {output_path}")

    return structured_blocks