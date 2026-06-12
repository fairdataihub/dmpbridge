"""Extract text blocks from a PDF using pdfplumber."""

import pdfplumber
from pathlib import Path
from typing import Union


def extract_blocks(pdf_path: Union[str, Path]) -> list[dict]:
    """
    Open a PDF and return a list of text-line blocks with coordinates,
    font info, and a placeholder 'label' field.

    Compatible with the dmpbridge viewer format.
    """
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = page.extract_text_lines(
                layout=True,
                strip_whitespace=True,
                return_chars=True,
            ) or []

            for line_order, line in enumerate(lines, start=1):
                text = line.get("text", "").strip()
                if not text:
                    continue

                chars = line.get("chars", [])

                # Deduplicated font names in order of appearance
                seen: set[str] = set()
                font_names: list[str] = []
                for c in chars:
                    fn = c.get("fontname", "")
                    if fn and fn not in seen:
                        seen.add(fn)
                        font_names.append(fn)

                sizes = [c["size"] for c in chars if c.get("size")]
                avg_font_size = sum(sizes) / len(sizes) if sizes else 0.0
                is_bold = any(_font_is_bold(fn) for fn in font_names)

                blocks.append({
                    "page": page_num,
                    "line_order": line_order,
                    "text": text,
                    "x0": round(float(line.get("x0", 0)), 2),
                    "top": round(float(line.get("top", 0)), 2),
                    "x1": round(float(line.get("x1", 0)), 2),
                    "bottom": round(float(line.get("bottom", 0)), 2),
                    "avg_font_size": round(avg_font_size, 2),
                    "font_names": font_names,
                    "is_bold": is_bold,
                    "label": None,
                })

    return blocks


def _font_is_bold(name: str) -> bool:
    n = name.lower()
    return "bold" in n or n.endswith(",b") or "-bold" in n or ",bold" in n
