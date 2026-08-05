"""pdfplumber utilities — text extraction, cleaning, and page image rendering.

These modules are used by :class:`~dmpbridge.extractors.PdfplumberExtractor`
and by visualisation tools.  Docling and LightOnOCR extraction live in
``dmpbridge/extractors/`` instead.

Sub-modules
-----------
pdfplumber_reader
    Line-level text extraction with full bbox + font metadata.
line_merger
    Joins wrapped lines back into paragraph-level blocks before classification.
text_cleaner
    Removes duplicate words/chars caused by layered PDF rendering (pdfplumber artefact).
page_images
    render_pages()     — clean page PNGs saved to disk; reusable across extractors.
    save_page_images() — page PNGs with per-label bounding-box overlays (pdfplumber blocks only).
"""
from .line_merger import merge_wrapped_lines
from .page_images import render_pages, save_page_images
from .pdfplumber_reader import extract_blocks
from .text_cleaner import clean_blocks, clean_repeated_words

__all__ = [
    "extract_blocks", "merge_wrapped_lines", "render_pages", "save_page_images",
    "clean_blocks", "clean_repeated_words",
]
