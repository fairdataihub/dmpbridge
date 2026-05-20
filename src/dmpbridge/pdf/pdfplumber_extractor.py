from pathlib import Path
import pdfplumber

from dmpbridge.processing.text_cleaner import normalize_text
from dmpbridge.utils.file_io import save_json, save_text
from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    """
    Return the root folder of the dmpbridge project.

    This file is located at:
    src/dmpbridge/pdf/pdfplumber_extractor.py

    parents[3] points back to:
    dmpbridge/
    """
    return Path(__file__).resolve().parents[3]


def extract_lines_with_pdfplumber(pdf_path: str | Path) -> list[dict]:
    """
    Extract line-level text and layout metadata from a PDF using pdfplumber.

    This function does NOT decide whether a line is a section, question, or answer.
    It only extracts raw line blocks with useful metadata, such as:

    - page number
    - line order
    - text
    - bounding box coordinates
    - average font size
    - font names
    - bold status

    The structure detection happens later in structure_detector.py.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    log(f"Extracting line-level text with pdfplumber: {pdf_path.name}")

    line_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            # Extract words with font metadata.
            # extra_attrs gives us font name and size, which help later
            # when detecting headings or section titles.
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,
                extra_attrs=["fontname", "size"]
            )

            # Group words into lines using their vertical position.
            # Words with the same rounded "top" value are treated as one line.
            lines = {}

            for word in words:
                top_key = round(word["top"], 1)

                if top_key not in lines:
                    lines[top_key] = []

                lines[top_key].append(word)

            # Process lines from top to bottom on the page.
            for line_order, top_key in enumerate(sorted(lines.keys()), start=1):

                # Sort words left to right within the same line.
                line_words = sorted(lines[top_key], key=lambda w: w["x0"])

                # Join words into one line of text and normalize minor issues.
                text = " ".join(w["text"] for w in line_words).strip()
                text = normalize_text(text)

                if not text:
                    continue

                # Collect font sizes for the line.
                font_sizes = [
                    w.get("size") for w in line_words
                    if w.get("size") is not None
                ]

                # Collect font names for the line.
                font_names = [
                    w.get("fontname") for w in line_words
                    if w.get("fontname") is not None
                ]

                # Average font size is useful for detecting headings later.
                avg_font_size = (
                    sum(font_sizes) / len(font_sizes)
                    if font_sizes else None
                )

                # Detect bold text based on font name.
                is_bold = any(
                    "bold" in font.lower()
                    for font in font_names
                    if font
                )

                # Save one structured line block.
                line_blocks.append({
                    "source_pdf": pdf_path.name,
                    "page": page_number,
                    "line_order": line_order,
                    "text": text,

                    # Layout coordinates
                    "x0": min(w["x0"] for w in line_words),
                    "top": min(w["top"] for w in line_words),
                    "x1": max(w["x1"] for w in line_words),
                    "bottom": max(w["bottom"] for w in line_words),

                    # Font/style metadata
                    "avg_font_size": avg_font_size,
                    "font_names": sorted(list(set(font_names))),
                    "is_bold": is_bold,

                    # Provenance
                    "extractor": "pdfplumber"
                })

    return line_blocks


def save_pdfplumber_outputs(pdf_path: str | Path) -> list[dict]:
    """
    Run pdfplumber extraction and save two outputs:

    1. Line-level JSON:
       data/pdfplumber_blocks/{pdf_name}.json

    2. Plain extracted text:
       data/extracted_text/{pdf_name}.txt

    The JSON file is used by the next step of the pipeline.
    The TXT file is mainly for human review/debugging.
    """

    pdf_path = Path(pdf_path)
    project_root = get_project_root()

    line_blocks = extract_lines_with_pdfplumber(pdf_path)

    output_json = project_root / "data" / "pdfplumber_extracted_blocks" / f"{pdf_path.stem}.json"
    output_txt = project_root / "data" / "pdfplumber_extracted_text" / f"{pdf_path.stem}.txt"
    output_md = project_root / "data" / "pdfplumber_extracted_markdown" / f"{pdf_path.stem}.md"

    save_json(line_blocks, output_json)

    # Create a readable plain-text version grouped by page.
    text_lines = []
    current_page = None

    for block in line_blocks:
        if block["page"] != current_page:
            current_page = block["page"]
            text_lines.append(f"\n\n<!-- Page {current_page} -->\n")

        text_lines.append(block["text"])

    save_text("\n".join(text_lines), output_txt)
    save_text("\n".join(text_lines), output_md)

    log(f"Saved line-level JSON: {output_json}")
    log(f"Saved extracted text: {output_txt}")
    log(f"Saved Markdown text: {output_md}")

    return line_blocks