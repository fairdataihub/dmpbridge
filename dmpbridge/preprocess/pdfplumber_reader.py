"""Extract text blocks from a PDF using pdfplumber.

Each page is scanned line by line.  Every text line becomes one block dict
containing the raw text, bounding-box coordinates, font metadata, and an empty
label field that is filled in later by the classifier.
"""
import re
from pathlib import Path
from typing import Union

import pdfplumber

from .text_cleaner import clean_blocks


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
    return clean_blocks(blocks)


def _line_to_blocks(line: dict, page_num: int, line_order: int) -> list[dict]:
    """Convert one pdfplumber line into one or two block dicts.

    1. Clean and deduplicate the text.
    2. Walk each character to collect font names, sizes, and bounding box.
    3. Detect bold→regular font transitions within the line.
    4. If the line starts with a bold segment ending in ':' followed by regular
       text, split into two blocks (bold label + regular answer) so the
       classifier can label them independently.
    """
    chars = line.get("chars", [])
    text  = _deduplicate_chars(line.get("text", "").strip())
    if not text:
        return []

    # Annotate each character with its bold flag
    annotated = []
    for c in chars:
        ch = c.get("text", "")
        if not ch:
            continue
        fn = c.get("fontname", "")
        annotated.append({
            "ch":     ch,
            "bold":   _font_is_bold(fn),
            "italic": _font_is_italic(fn),
            "fn":     fn,
            "size":   c.get("size", 0),
            "x0":     c.get("x0", 0),
            "x1":     c.get("x1", 0),
            "top":    c.get("top", 0),
            "bottom": c.get("bottom", 0),
        })

    if not annotated:
        return []

    # Detect bold→regular transition: line starts bold, then switches to regular
    split_idx = _find_bold_to_regular_split(annotated)

    if split_idx is not None:
        bold_chars    = annotated[:split_idx]
        regular_chars = annotated[split_idx:]
        # Use the original line["text"] (which includes spaces) as the source of
        # truth for text, and find the split position by counting non-space chars.
        bold_text, regular_text = _split_at_nonspace_count(text, split_idx)
        bold_text    = _deduplicate_chars(bold_text)
        regular_text = _deduplicate_chars(regular_text)
        if bold_text and regular_text:
            return [
                _make_block(bold_chars,    bold_text,    page_num, line_order, is_bold=True),
                _make_block(regular_chars, regular_text, page_num, line_order, is_bold=False),
            ]

    # Default: single block
    is_bold   = annotated[0]["bold"]
    is_italic = annotated[0]["italic"]
    return [_make_block(annotated, text, page_num, line_order,
                        is_bold=is_bold, is_italic=is_italic)]


def _make_block(
    chars: list[dict],
    text: str,
    page_num: int,
    line_order: int,
    is_bold: bool = False,
    is_italic: bool = False,
) -> dict:
    """Build a block dict from a list of annotated character dicts."""
    seen: set[str] = set()
    font_names: list[str] = []
    sizes: list[float] = []
    x0 = x1 = top = bottom = None

    for c in chars:
        fn = c.get("fn", "")
        if fn and fn not in seen:
            seen.add(fn)
            font_names.append(fn)
        if c.get("size"):
            sizes.append(c["size"])
        cx0, cx1 = c.get("x0", 0), c.get("x1", 0)
        ct,  cb  = c.get("top", 0), c.get("bottom", 0)
        if x0 is None or cx0 < x0:
            x0 = cx0
        if x1 is None or cx1 > x1:
            x1 = cx1
        if top is None or ct < top:
            top = ct
        if bottom is None or cb > bottom:
            bottom = cb

    return {
        "page":          page_num,
        "line_order":    line_order,
        "text":          text,
        "x0":            round(float(x0 or 0), 2),
        "top":           round(float(top or 0), 2),
        "x1":            round(float(x1 or 0), 2),
        "bottom":        round(float(bottom or 0), 2),
        "avg_font_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        "font_names":    font_names,
        "is_bold":       is_bold,
        "is_italic":     is_italic,
        "label":         None,
    }


def _split_at_nonspace_count(text: str, n: int) -> tuple[str, str]:
    """Split *text* after the n-th non-whitespace character.

    pdfplumber's line["text"] contains spaces; individual char["text"] entries
    do not.  We count non-space chars in the full text to find the split point
    that matches the char-level split index.
    """
    count = 0
    for i, ch in enumerate(text):
        if ch != " ":
            count += 1
        if count == n:
            return text[: i + 1].strip(), text[i + 1 :].strip()
    return text, ""


def _find_bold_to_regular_split(annotated: list[dict]) -> int | None:
    """Return the character index where a bold prefix ends and regular text begins.

    Conditions:
    - Line must start with bold text (ignoring leading spaces)
    - Bold segment must end with ':' or ': '
    - Regular segment must be non-empty
    - Bold segment must be short (≤ 60 chars) — avoids splitting long bold headings
    """
    # Find first non-space character
    start = next((i for i, c in enumerate(annotated) if c["ch"].strip()), None)
    if start is None or not annotated[start]["bold"]:
        return None

    # Walk forward while still bold
    split = None
    bold_text = ""
    for i in range(start, len(annotated)):
        c = annotated[i]
        if not c["bold"] and c["ch"].strip():
            # Found first non-bold, non-space character
            split = i
            break
        bold_text += c["ch"]

    if split is None:
        return None  # Entire line is bold

    bold_stripped = bold_text.strip()
    # Bold segment must end with ':' and be reasonably short
    if not bold_stripped.endswith(":"):
        return None
    if len(bold_stripped) > 60:
        return None
    # Regular part must have real content
    regular_text = "".join(c["ch"] for c in annotated[split:]).strip()
    if not regular_text:
        return None

    return split


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
