from pathlib import Path
from typing import Dict, Any

from docling.document_converter import DocumentConverter

from dmpbridge.utils.file_io import save_text, save_json
from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def extract_with_docling(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Extract a PDF using Docling.

    Returns:
    - markdown text
    - raw Docling document dictionary if available
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log(f"Extracting PDF with Docling: {pdf_path.name}")

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    markdown_text = result.document.export_to_markdown()

    try:
        docling_json = result.document.export_to_dict()
    except Exception:
        docling_json = {
            "warning": "Could not export Docling document to dict"
        }

    return {
        "source_pdf": pdf_path.name,
        "extractor": "docling",
        "markdown": markdown_text,
        "docling_json": docling_json,
    }


def save_docling_outputs(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Save Docling Markdown and JSON outputs.
    """

    pdf_path = Path(pdf_path)
    project_root = get_project_root()

    result = extract_with_docling(pdf_path)

    markdown_path = (
        project_root
        / "data"
        / "docling_markdown"
        / f"{pdf_path.stem}.md"
    )

    json_path = (
        project_root
        / "data"
        / "docling_json"
        / f"{pdf_path.stem}.json"
    )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    save_text(result["markdown"], markdown_path)
    save_json(result["docling_json"], json_path)

    log(f"Saved Docling Markdown: {markdown_path}")
    log(f"Saved Docling JSON: {json_path}")

    return result