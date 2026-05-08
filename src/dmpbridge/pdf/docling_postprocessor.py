from pathlib import Path
from typing import List, Dict, Any

from dmpbridge.utils.file_io import load_text, save_json
from dmpbridge.utils.logger import log



def markdown_to_structured_blocks(
    markdown_text: str,
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:
    structured_blocks = []
    line_order = 1
    seen_title = False

    for line in markdown_text.splitlines():
        text = line.strip()

        if not text:
            continue

        label = "content"

        if text.startswith("#"):
            heading_level = len(text) - len(text.lstrip("#"))
            heading_text = text.lstrip("#").strip()

            if not heading_text:
                continue

            text = heading_text

            if heading_level == 1 and not seen_title:
                label = "document_title"
                seen_title = True
            elif heading_level in [1, 2]:
                label = "section"
            else:
                label = "subsection"

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

    markdown_text = load_text(markdown_path)

    structured_blocks = markdown_to_structured_blocks(
        markdown_text=markdown_text,
        source_pdf=source_pdf
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(structured_blocks, output_path)

    log(f"Saved Docling structured blocks: {output_path}")

    return structured_blocks