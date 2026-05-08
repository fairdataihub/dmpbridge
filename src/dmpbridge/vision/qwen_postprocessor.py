from pathlib import Path
from typing import List, Dict, Any

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.utils.logger import log


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def is_generic_document_heading(text: str) -> bool:
    """
    General document-level headings that should not become narrative sections.
    """
    normalized = text.strip().lower()

    generic_titles = {
        "data management plan",
        "data management and sharing plan",
        "dmp",
    }

    return normalized in generic_titles


def is_likely_body_text(text: str) -> bool:
    """
    General filter to remove paragraph/body text accidentally returned as headings.

    This avoids sample-specific rules.
    """

    text = text.strip()

    if not text:
        return True

    words = text.split()

    # Long text is likely body text, not a heading.
    if len(words) > 12:
        return True

    # Sentence-like text is likely body text.
    if text.endswith(".") and len(words) > 6:
        return True

    # Headings are usually short and do not contain many commas.
    if text.count(",") >= 2 and len(words) > 6:
        return True

    return False


def add_block(
    structured_blocks: List[Dict[str, Any]],
    source_pdf: str | None,
    page_number: int,
    line_order: int,
    text: str,
    label: str
) -> int:
    structured_blocks.append({
        "source_pdf": source_pdf,
        "page": page_number,
        "line_order": line_order,
        "text": text,
        "label": label,
        "document_format": "qwen_vl",
        "extractor": "qwen_vl"
    })

    return line_order + 1


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

    General cleanup:
    - generic DMP headings are not treated as sections
    - long paragraph-like text is removed
    - document title is saved once
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
            if not is_generic_document_heading(document_title):
                global_order = add_block(
                    structured_blocks=structured_blocks,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    line_order=global_order,
                    text=document_title,
                    label="document_title"
                )

                seen_document_title = True

        sections = page_result.get("sections", [])

        if not isinstance(sections, list):
            continue

        for section in sections:
            if not isinstance(section, dict):
                continue

            section_title = clean_text(section.get("title"))

            skip_section = (
                not section_title
                or is_generic_document_heading(section_title)
                or is_likely_body_text(section_title)
            )

            if not skip_section:
                global_order = add_block(
                    structured_blocks=structured_blocks,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    line_order=global_order,
                    text=section_title,
                    label="section"
                )

            subsections = section.get("subsections", [])

            if not isinstance(subsections, list):
                continue

            for subsection in subsections:
                if isinstance(subsection, dict):
                    subsection_title = clean_text(subsection.get("title"))
                else:
                    subsection_title = clean_text(subsection)

                if (
                    not subsection_title
                    or is_generic_document_heading(subsection_title)
                    or is_likely_body_text(subsection_title)
                ):
                    continue

                # If the parent section was skipped, promote the subsection to section.
                # This handles cases where Qwen puts real topic headings under a generic
                # wrapper like "DATA MANAGEMENT PLAN".
                label = "subsection" if not skip_section else "section"

                global_order = add_block(
                    structured_blocks=structured_blocks,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    line_order=global_order,
                    text=subsection_title,
                    label=label
                )

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