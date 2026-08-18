"""pdfplumber utilities — whole-document text extraction and page image rendering.

These modules are used by :class:`~dmpbridge.extractors.PdfplumberExtractor`
and by visualisation tools.

Sub-modules
-----------
pdfplumber_reader
    extract_text_for_llm() — whole document as one text string, with
    **bold**/_italic_ markers standing in for visual emphasis.
page_images
    render_pages()     — clean page PNGs saved to disk.
    save_page_images() — page PNGs with per-label bounding-box overlays (blocks that carry bbox data only;
                          none of the current pipeline's output does, so this draws no boxes today).
"""
from .page_images import render_pages, save_page_images
from .pdfplumber_reader import extract_text_for_llm

__all__ = ["extract_text_for_llm", "render_pages", "save_page_images"]
