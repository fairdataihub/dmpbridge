"""Extract text blocks from a PDF using pdfplumber.

Each page is scanned line by line.  Every text line becomes one block dict
containing the raw text, bounding-box coordinates, font metadata, and an empty
label field that is filled in later by the classifier.
"""
import re
from pathlib import Path
from typing import Union

import pdfplumber


def extract_blocks(pdf_path: Union[str, Path]) -> list[dict]:
    """Open the PDF and turn every text line into a block dict.

    1. Open the PDF with pdfplumber.
    2. For each page, extract all text lines with full character-level layout info.
    3. Convert each line into a block with text, position, font info, and an empty label.
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
                blocks.extend(_line_to_blocks(line, page_num, line_order))
    return blocks


def _line_to_blocks(line: dict, page_num: int, line_order: int) -> list[dict]:
    """Convert one pdfplumber line into a single block dict.

    1. Clean and deduplicate the text.
    2. Walk each character to collect font names, sizes, and bounding box.
    3. Determine bold/italic from the first non-whitespace character's font name.
    """
    chars = line.get("chars", [])
    text  = _deduplicate_chars(line.get("text", "").strip())
    if not text:
        return []

    seen: set[str] = set()
    font_names: list[str] = []
    first_bold = first_italic = None
    sizes: list[float] = []
    x0 = x1 = top = bottom = None

    for c in chars:
        fn = c.get("fontname", "")
        if fn and fn not in seen:
            seen.add(fn)
            font_names.append(fn)
        if c.get("size"):
            sizes.append(c["size"])
        cx0, cx1 = c.get("x0", 0), c.get("x1", 0)
        ct, cb   = c.get("top", 0), c.get("bottom", 0)
        if x0 is None or cx0 < x0:  x0 = cx0
        if x1 is None or cx1 > x1:  x1 = cx1
        if top    is None or ct < top:     top    = ct
        if bottom is None or cb > bottom:  bottom = cb
        if first_bold is None and c.get("text", "").strip():
            first_bold   = _font_is_bold(fn)
            first_italic = _font_is_italic(fn)

    return [{
        "page":          page_num,
        "line_order":    line_order,
        "text":          text,
        "x0":            round(float(x0 or 0), 2),
        "top":           round(float(top or 0), 2),
        "x1":            round(float(x1 or 0), 2),
        "bottom":        round(float(bottom or 0), 2),
        "avg_font_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        "font_names":    font_names,
        "is_bold":       bool(first_bold),
        "is_italic":     bool(first_italic),
        "label":         None,
    }]


# ── Font helpers ──────────────────────────────────────────────────────────────

def _font_is_bold(name: str) -> bool:
    n = name.lower()
    return "bold" in n or n.endswith(",b") or "-bold" in n or ",bold" in n


def _font_is_italic(name: str) -> bool:
    n = name.lower()
    return "italic" in n or "oblique" in n or n.endswith(",i") or "-italic" in n


def _deduplicate_chars(text: str) -> str:
    """Fix doubled characters caused by layered PDF text rendering.

    Some PDFs render text twice (bold shadow over regular), so pdfplumber
    returns 'HHeelllloo'.  Detects this pattern and collapses the duplicates.
    """
    stripped = text.replace(" ", "")
    if len(stripped) < 4:
        return text
    pairs = sum(
        1 for i in range(0, len(stripped) - 1, 2)
        if stripped[i] == stripped[i + 1]
    )
    if pairs / (len(stripped) / 2) > 0.7:
        return re.sub(r"(.)\1", r"\1", text)
    return text
