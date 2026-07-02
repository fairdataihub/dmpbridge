"""PDF preprocessing — text extraction and page image rendering.

Sub-modules
-----------
pdfplumber_reader
    Line-level text extraction using pdfplumber.
page_images
    render_pages()     — clean page PNGs saved to disk; reusable across models.
    save_page_images() — pages with colored bounding-box overlays for the viewer.
"""
from .page_images import render_pages, save_page_images
from .pdfplumber_reader import extract_blocks
from .text_cleaner import clean_blocks, clean_repeated_words

__all__ = ["extract_blocks", "render_pages", "save_page_images", "clean_blocks", "clean_repeated_words"]
