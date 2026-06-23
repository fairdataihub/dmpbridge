"""Extract text blocks from a PDF using pdfplumber."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Union

import pdfplumber

# RGBA colors per label — (stroke, fill)
_LABEL_STYLE: dict[str, tuple[tuple, tuple]] = {
    "title":               ((220,  38,  38, 220), (220,  38,  38, 30)),
    "section.title":       ((34,  197,  94, 220), (34,  197,  94, 25)),
    "section.description": ((59,  130, 246, 200), (59,  130, 246, 20)),
    "question.text":       ((245, 158,  11, 220), (245, 158,  11, 25)),
    "answer.text":         ((168,  85, 247, 180), (168,  85, 247, 15)),
}


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
                for block in _line_to_blocks(line, page_num, line_order):
                    blocks.append(block)

    return blocks


def _line_to_blocks(line: dict, page_num: int, line_order: int) -> list[dict]:
    """
    Convert one pdfplumber line into a single block.

    Bold/italic is determined by majority vote over the non-whitespace chars
    so that lines with a mixed-style leading token (e.g. bold question header
    followed by regular text on the same line) are still classified correctly.
    """
    chars = line.get("chars", [])
    text = _deduplicate_chars(line.get("text", "").strip())
    if not text:
        return []

    seen: set[str] = set()
    font_names: list[str] = []
    bold_count = italic_count = total = 0
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
        if x0 is None or cx0 < x0: x0 = cx0
        if x1 is None or cx1 > x1: x1 = cx1
        if top is None or ct < top: top = ct
        if bottom is None or cb > bottom: bottom = cb
        if c.get("text", "").strip():
            total += 1
            if _font_is_bold(fn):   bold_count   += 1
            if _font_is_italic(fn): italic_count += 1

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
        "is_bold":       bold_count   > (total / 2 if total else 1),
        "is_italic":     italic_count > (total / 2 if total else 1),
        "label":         None,
    }]


def save_page_images(
    pdf_path: Union[str, Path],
    blocks: list[dict],
    output_dir: Union[str, Path] = "pdfplumber",
    resolution: int = 150,
) -> list[Path]:
    """Render each PDF page as a PNG with block bounding boxes overlaid.

    Requires Pillow and a pdfplumber image backend (wand / ImageMagick).
    Install: pip install Pillow wand
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_blocks = [b for b in blocks if b["page"] == page_num]

            try:
                im = page.to_image(resolution=resolution)
            except Exception as exc:
                raise RuntimeError(
                    f"pdfplumber could not render page {page_num} as an image.\n"
                    "Ensure Pillow is installed: pip install Pillow\n"
                    f"Details: {exc}"
                ) from exc

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
    n = name.lower()
    return "bold" in n or n.endswith(",b") or "-bold" in n or ",bold" in n


def _font_is_italic(name: str) -> bool:
    n = name.lower()
    return "italic" in n or "oblique" in n or n.endswith(",i") or "-italic" in n


def _deduplicate_chars(text: str) -> str:
    """Collapse doubled characters caused by layered PDF text rendering.

    Some PDFs render text twice (bold shadow over regular), causing pdfplumber
    to return "HHeelllloo" instead of "Hello". Detects this by checking whether
    most consecutive character pairs in the non-space content are identical,
    then applies (.)\1 -> \1 substitution.
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
