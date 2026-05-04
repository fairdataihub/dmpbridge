from pathlib import Path
import pdfplumber

from dmpbridge.utils.file_io import save_json, save_text
from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    """
    Returns the main dmpbridge project folder.
    Current file is:
    src/dmpbridge/pdf/pdfplumber_extractor.py

    parents[3] = dmpbridge/
    """
    return Path(__file__).resolve().parents[3]


def extract_lines_with_pdfplumber(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    log(f"Extracting line-level text with pdfplumber: {pdf_path.name}")

    line_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,
                extra_attrs=["fontname", "size"]
            )

            lines = {}

            for word in words:
                top_key = round(word["top"], 1)

                if top_key not in lines:
                    lines[top_key] = []

                lines[top_key].append(word)

            for line_order, top_key in enumerate(sorted(lines.keys()), start=1):
                line_words = sorted(lines[top_key], key=lambda w: w["x0"])

                text = " ".join(w["text"] for w in line_words).strip()

                if not text:
                    continue

                font_sizes = [
                    w.get("size") for w in line_words
                    if w.get("size") is not None
                ]

                font_names = [
                    w.get("fontname") for w in line_words
                    if w.get("fontname") is not None
                ]

                avg_font_size = (
                    sum(font_sizes) / len(font_sizes)
                    if font_sizes else None
                )

                is_bold = any(
                    "bold" in font.lower()
                    for font in font_names
                    if font
                )

                line_blocks.append({
                    "source_pdf": pdf_path.name,
                    "page": page_number,
                    "line_order": line_order,
                    "text": text,
                    "x0": min(w["x0"] for w in line_words),
                    "top": min(w["top"] for w in line_words),
                    "x1": max(w["x1"] for w in line_words),
                    "bottom": max(w["bottom"] for w in line_words),
                    "avg_font_size": avg_font_size,
                    "font_names": list(set(font_names)),
                    "is_bold": is_bold,
                    "extractor": "pdfplumber"
                })

    return line_blocks


def save_pdfplumber_outputs(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    project_root = get_project_root()

    line_blocks = extract_lines_with_pdfplumber(pdf_path)

    output_json = project_root / "data" / "pdfplumber_blocks" / f"{pdf_path.stem}.json"
    output_txt = project_root / "data" / "extracted_text" / f"{pdf_path.stem}.txt"

    save_json(line_blocks, output_json)

    text_lines = []
    current_page = None

    for block in line_blocks:
        if block["page"] != current_page:
            current_page = block["page"]
            text_lines.append(f"\n\n<!-- Page {current_page} -->\n")

        text_lines.append(block["text"])

    save_text("\n".join(text_lines), output_txt)

    log(f"Saved line-level JSON: {output_json}")
    log(f"Saved extracted text: {output_txt}")

    return line_blocks