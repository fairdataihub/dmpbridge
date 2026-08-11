"""Extract text blocks from a PDF using pdfplumber.

Every text line pdfplumber reports becomes exactly one block, carrying that
line's text along with its bounding box and font metadata.

Nothing here changes the *structure* of what pdfplumber returns.  Two earlier
steps that did — line merging, and a bold-to-regular split that turned one line
into two blocks — have been removed, so the classifier sees pdfplumber's own
segmentation.

Text repair is kept, because it fixes the reader misreading the page rather than
restructuring it:

* ``_deduplicate_chars`` collapses ``HHeelllloo`` back to ``Hello``, which some
  PDFs produce by rendering text twice in offset layers;
* ``clean_blocks`` removes the word-level form of the same artefact
  (``Roles Roles and and`` → ``Roles and``) and collapses excess whitespace.

Both only ever remove duplication introduced by rendering; neither joins,
splits, or reorders blocks.
"""
import re
from pathlib import Path
from typing import Union

import pdfplumber

from .text_cleaner import clean_blocks


def extract_blocks(pdf_path: Union[str, Path]) -> list[dict]:
    """Open the PDF and turn every text line into one block dict.

    1. Open the PDF with pdfplumber.
    2. For each page, extract all text lines with character-level layout info.
    3. Convert each line into a single block: text, position, font metadata.
    4. Repair rendering artefacts in the text — no structural change.

    No ``label`` key is written. Extraction knows nothing about classification;
    the strategy adds ``label`` and ``confidence`` at stage 2.
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
                block = _line_to_block(line, page_num, line_order)
                if block is not None:
                    blocks.append(block)
    return clean_blocks(blocks)


def _line_to_block(line: dict, page_num: int, line_order: int) -> dict | None:
    """Convert one pdfplumber line into one block, or None when it is empty."""
    text = _deduplicate_chars(line.get("text", "").strip())
    if not text:
        return None

    chars = [c for c in line.get("chars", []) if c.get("text")]
    font_names: list[str] = []
    sizes: list[float] = []
    for c in chars:
        fn = c.get("fontname", "")
        if fn and fn not in font_names:
            font_names.append(fn)
        if c.get("size"):
            sizes.append(c["size"])

    first = chars[0].get("fontname", "") if chars else ""

    return {
        "page":          page_num,
        "line_order":    line_order,
        "text":          text,
        "x0":            round(float(line.get("x0", 0)), 2),
        "top":           round(float(line.get("top", 0)), 2),
        "x1":            round(float(line.get("x1", 0)), 2),
        "bottom":        round(float(line.get("bottom", 0)), 2),
        "avg_font_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        "font_names":    font_names,
        "is_bold":       _font_is_bold(first),
        "is_italic":     _font_is_italic(first),
    }


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
    returns 'HHeelllloo'.  Detects that pattern and collapses the duplicates.
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
