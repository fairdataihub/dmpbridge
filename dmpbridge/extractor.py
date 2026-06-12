"""Extract text blocks from a PDF using pdfplumber."""

import pdfplumber
from collections import defaultdict
from pathlib import Path
from typing import Union

# RGBA colors per label — (stroke, fill)
_LABEL_STYLE: dict[str, tuple[tuple, tuple]] = {
    "document_title": ((139, 92, 246, 220), (139, 92, 246, 30)),
    "section":        ((245, 158, 11,  220), (245, 158, 11,  25)),
    "subsection":     ((20,  184, 166, 200), (20,  184, 166, 20)),
    "content":        ((88,  166, 255, 120), (88,  166, 255, 15)),
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
                    "Install the required backend:  pip install Pillow wand\n"
                    "wand also needs ImageMagick:   https://imagemagick.org/script/download.php\n"
                    f"Details: {exc}"
                ) from exc

            by_label: dict[str, list[dict]] = defaultdict(list)
            for b in page_blocks:
                by_label[b.get("label") or "content"].append(b)

            for label, group in by_label.items():
                stroke, fill = _LABEL_STYLE.get(label, _LABEL_STYLE["content"])
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
