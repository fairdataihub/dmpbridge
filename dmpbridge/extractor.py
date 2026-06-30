"""Extract text blocks from a PDF using pdfplumber."""

# What this file does — step by step:
#   Step 1 — open the PDF and loop through every page
#   Step 2 — extract every text line with full character-level detail (font, size, position)
#   Step 3 — convert each line into a block dict (text, bounding box, bold, italic, font size)
#   Step 4 — fix doubled characters that some PDFs produce from layered text rendering
#   Step 5 — return a flat list of block dicts with label = None, ready for the LLM

import re
from collections import defaultdict
from pathlib import Path
from typing import Union

import pdfplumber

# Colors used to draw bounding boxes on page images — one color per label.
# Each entry is (stroke_color, fill_color) as RGBA tuples.
_LABEL_STYLE: dict[str, tuple[tuple, tuple]] = {
    "title":               ((220,  38,  38, 220), (220,  38,  38, 30)),
    "section.title":       ((34,  197,  94, 220), (34,  197,  94, 25)),
    "section.description": ((59,  130, 246, 200), (59,  130, 246, 20)),
    "question.text":       ((245, 158,  11, 220), (245, 158,  11, 25)),
    "answer.text":         ((168,  85, 247, 180), (168,  85, 247, 15)),
}


def extract_blocks(pdf_path: Union[str, Path]) -> list[dict]:
    """Open the PDF and turn every text line into a block dict. 1. Open the PDF with pdfplumber. 2. For each page, extract all text lines with layout info. 3. Convert each line into a block with text, position, font info, and an empty label."""
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract lines with full character-level detail so we can read font names and sizes.
            lines = page.extract_text_lines(
                layout=True,
                strip_whitespace=True,
                return_chars=True,
            ) or []

            for line_order, line in enumerate(lines, start=1):
                for block in _line_to_blocks(line, page_num, line_order):
                    blocks.append(block)

    return blocks


def _line_to_blocks(line: dict, page_num: int, line_order: int) -> list[dict]:
    """Convert one pdfplumber line into a single block dict. 1. Clean and deduplicate the text. 2. Walk each character to collect font names, sizes, and bounding box. 3. Determine bold/italic from the first non-whitespace character's font name. 4. Return a single-item list with the assembled block."""
    chars = line.get("chars", [])
    # Clean the text and collapse any doubled characters from layered PDF rendering.
    text = _deduplicate_chars(line.get("text", "").strip())
    if not text:
        return []

    seen: set[str] = set()
    font_names: list[str] = []
    first_bold = first_italic = None
    sizes: list[float] = []
    x0 = x1 = top = bottom = None

    for c in chars:
        fn = c.get("fontname", "")
        # Collect unique font names used in this line (there can be more than one).
        if fn and fn not in seen:
            seen.add(fn)
            font_names.append(fn)
        if c.get("size"):
            sizes.append(c["size"])
        # Track the overall bounding box of the line across all characters.
        cx0, cx1 = c.get("x0", 0), c.get("x1", 0)
        ct, cb   = c.get("top", 0), c.get("bottom", 0)
        if x0 is None or cx0 < x0: x0 = cx0
        if x1 is None or cx1 > x1: x1 = cx1
        if top is None or ct < top: top = ct
        if bottom is None or cb > bottom: bottom = cb
        # Use the first visible (non-whitespace) character to decide bold/italic for the whole line.
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
        "label":         None,  # filled in later by the LLM classifier
    }]


def save_page_images(
    pdf_path: Union[str, Path],
    blocks: list[dict],
    output_dir: Union[str, Path] = "pdfplumber",
    resolution: int = 150,
) -> list[Path]:
    """Render each page as a PNG with colored bounding boxes drawn over each labeled block. 1. Open the PDF. 2. For each page, render it as an image. 3. Group blocks by label and draw colored rectangles. 4. Save the image and return the list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Only keep blocks that belong to this page.
            page_blocks = [b for b in blocks if b["page"] == page_num]

            try:
                im = page.to_image(resolution=resolution)
            except Exception as exc:
                raise RuntimeError(
                    f"pdfplumber could not render page {page_num} as an image.\n"
                    "Ensure Pillow is installed: pip install Pillow\n"
                    f"Details: {exc}"
                ) from exc

            # Group blocks by label so we can draw all boxes of the same label in one color.
            by_label: dict[str, list[dict]] = defaultdict(list)
            for b in page_blocks:
                by_label[b.get("label") or "answer.text"].append(b)

            for label, group in by_label.items():
                stroke, fill = _LABEL_STYLE.get(label, _LABEL_STYLE["answer.text"])
                rects = [
                    {"x0": b["x0"], "top": b["top"], "x1": b["x1"], "bottom": b["bottom"]}
                    for b in group
                ]
                im.draw_rects(rects, stroke=stroke, stroke_width=1, fill=fill)

            out = output_dir / f"page_{page_num:03d}.png"
            im.save(out)
            saved.append(out)

    return saved


def _font_is_bold(name: str) -> bool:
    # Check common bold font name patterns used across different PDF generators.
    n = name.lower()
    return "bold" in n or n.endswith(",b") or "-bold" in n or ",bold" in n


def _font_is_italic(name: str) -> bool:
    # Check common italic/oblique font name patterns.
    n = name.lower()
    return "italic" in n or "oblique" in n or n.endswith(",i") or "-italic" in n


def _deduplicate_chars(text: str) -> str:
    """Fix doubled characters caused by layered PDF text rendering. Some PDFs render text twice (bold shadow over regular), so pdfplumber returns 'HHeelllloo' instead of 'Hello'. Detects this by checking if most consecutive character pairs are identical, then collapses them."""
    stripped = text.replace(" ", "")
    if len(stripped) < 4:
        return text
    # Count how many consecutive char pairs are duplicates.
    pairs = sum(
        1 for i in range(0, len(stripped) - 1, 2)
        if stripped[i] == stripped[i + 1]
    )
    # If more than 70% of pairs are duplicates, collapse every repeated character.
    if pairs / (len(stripped) / 2) > 0.7:
        return re.sub(r"(.)\1", r"\1", text)
    return text
