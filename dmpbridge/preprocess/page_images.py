"""Render PDF pages as PNG images.

Two modes:
- render_pages()      — clean page renders saved to disk.
- save_page_images()  — page PNGs with per-label bounding-box overlays. Blocks
                        without bbox data are silently skipped rather than
                        erroring — and pdfplumber's whole-document output
                        carries none, so this currently draws no boxes at all;
                        the page image is still rendered. Kept for a future
                        extractor that does carry bbox data.
"""
from collections import defaultdict
from pathlib import Path
from typing import Union

import pdfplumber

from ..utils import ExtractionError

# One (stroke, fill) RGBA pair per label.
_LABEL_STYLE: dict[str, tuple[tuple, tuple]] = {
    "title":               ((220,  38,  38, 220), (220,  38,  38, 30)),
    "section.title":       ((34,  197,  94, 220), (34,  197,  94, 25)),
    "section.description": ((59,  130, 246, 200), (59,  130, 246, 20)),
    "question.text":       ((245, 158,  11, 220), (245, 158,  11, 25)),
    "answer.text":         ((168,  85, 247, 180), (168,  85, 247, 15)),
}


def save_page_images(
    pdf_path: Union[str, Path],
    blocks: list[dict],
    output_dir: Union[str, Path] = "pdfplumber",
    resolution: int = 150,
) -> list[Path]:
    """Render each page as a PNG with colored bounding boxes per label.

    1. Open the PDF.
    2. For each page, render it as an image at *resolution* DPI.
    3. Group blocks by label and draw colored rectangles.
    4. Save the image and return the list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_blocks = [b for b in blocks if b.get("page") == page_num]

            try:
                im = page.to_image(resolution=resolution)
            except Exception as exc:
                raise ExtractionError(
                    f"pdfplumber could not render page {page_num} as an image.\n"
                    "Ensure Pillow is installed: pip install Pillow\n"
                    f"Details: {exc}"
                ) from exc

            # Only draw boxes for blocks that carry bbox data — pdfplumber's
            # whole-document output has none, so this always skips today.
            by_label: dict[str, list[dict]] = defaultdict(list)
            for b in page_blocks:
                if b.get("x0") is None:
                    continue
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


def render_pages(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    resolution: int = 150,
) -> list[Path]:
    """Render each PDF page as a plain PNG with no overlays and save to disk.

    Pages that already exist on disk are skipped, so this is safe to call
    repeatedly and across different model experiments.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    output_dir:
        Directory where page PNGs are written (``page_001.png``, ``page_002.png``, …).
    resolution:
        Render DPI (default 150 — good quality, reasonable file size).

    Returns
    -------
    list[Path]
        Paths of all page images (existing + newly rendered), in page order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            out = output_dir / f"page_{page_num:03d}.png"
            if not out.exists():
                try:
                    img = page.to_image(resolution=resolution)
                except Exception as exc:
                    raise ExtractionError(
                        f"pdfplumber could not render page {page_num} of {Path(pdf_path).name}.\n"
                        "Ensure Pillow is installed: pip install Pillow\n"
                        f"Details: {exc}"
                    ) from exc
                img.original.save(out, format="PNG")
            saved.append(out)

    return saved
