"""pdfplumber utilities — text extraction, cleaning, and page image rendering.

These modules are used by :class:`~dmpbridge.extractors.PdfplumberExtractor`
and by visualisation tools.  Docling and LightOnOCR extraction live in
``dmpbridge/extractors/`` instead.

Sub-modules
-----------
pdfplumber_reader
    One block per PDF text line, with bbox + font metadata.
text_cleaner
    Removes duplicate words/chars caused by layered PDF rendering (pdfplumber artefact).
page_images
    render_pages()     — clean page PNGs saved to disk; reusable across extractors.
    save_page_images() — page PNGs with per-label bounding-box overlays (pdfplumber blocks only).
"""
from .page_images import render_pages, save_page_images
from .pdfplumber_reader import extract_blocks
from .text_cleaner import clean_blocks, clean_repeated_words

__all__ = [
    "extract_blocks", "render_pages", "save_page_images",
    "clean_blocks", "clean_repeated_words",
]
