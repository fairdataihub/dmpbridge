from pathlib import Path
from typing import List, Dict, Any

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.utils.logger import log


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def convert_qwen_output_to_structured_blocks(
    qwen_results: List[Dict[str, Any]] | Dict[str, Any],
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:
    """
    Convert hierarchical Qwen2-VL output into the same structured_blocks format
    used by the rule-based pipeline.

    Expected Qwen page-level format:
    {
      "document_title": null,
      "sections": [
        {
          "title": "Section heading",
          "subsections": [
            {"title": "Subsection heading"}
          ]
        }
      ]
    }

    Output format:
    [
      {"text": "...", "label": "document_title"},
      {"text": "...", "label": "section"},
      {"text": "...", "label": "subsection"}
    ]
    """

    if isinstance(qwen_results, dict):
        qwen_results = [qwen_results]

    structured_blocks = []
    global_order = 1
    seen_document_title = False

    for page_index, page_result in enumerate(qwen_results, start=1):
        page_number = page_result.get("page", page_index)

        document_title = clean_text(page_result.get("document_title"))

        if document_title and not seen_document_title:
            structured_blocks.append({
                "source_pdf": source_pdf,
                "page": page_number,
                "line_order": global_order,
                "text": document_title,
                "label": "document_title",
                "document_format": "qwen_vl",
                "extractor": "qwen_vl"
            })

            global_order += 1
            seen_document_title = True

        sections = page_result.get("sections", [])

        if not isinstance(sections, list):
            continue

        for section in sections:
            if not isinstance(section, dict):
                continue

            section_title = clean_text(section.get("title"))

            if section_title:
                structured_blocks.append({
                    "source_pdf": source_pdf,
                    "page": page_number,
                    "line_order": global_order,
                    "text": section_title,
                    "label": "section",
                    "document_format": "qwen_vl",
                    "extractor": "qwen_vl"
                })

                global_order += 1

            subsections = section.get("subsections", [])

            if not isinstance(subsections, list):
                continue

            for subsection in subsections:
                if isinstance(subsection, dict):
                    subsection_title = clean_text(subsection.get("title"))
                else:
                    subsection_title = clean_text(subsection)

                if subsection_title:
                    structured_blocks.append({
                        "source_pdf": source_pdf,
                        "page": page_number,
                        "line_order": global_order,
                        "text": subsection_title,
                        "label": "subsection",
                        "document_format": "qwen_vl",
                        "extractor": "qwen_vl"
                    })

                    global_order += 1

    return structured_blocks


def save_qwen_structured_blocks(
    qwen_output_path: str | Path,
    output_path: str | Path,
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:
    """
    Load hierarchical Qwen JSON output, convert it to structured_blocks,
    and save the converted result.
    """

    qwen_output_path = Path(qwen_output_path)
    output_path = Path(output_path)

    qwen_results = load_json(qwen_output_path)

    structured_blocks = convert_qwen_output_to_structured_blocks(
        qwen_results=qwen_results,
        source_pdf=source_pdf
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(structured_blocks, output_path)

    log(f"Saved Qwen structured blocks: {output_path}")

    return structured_blocks