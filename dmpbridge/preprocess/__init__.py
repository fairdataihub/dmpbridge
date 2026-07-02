"""PDF preprocessing — text extraction and page image rendering.

Sub-modules
-----------
pdfplumber_reader
    Line-level text extraction using pdfplumber.
page_images
    Render PDF pages as PNG images with labeled bounding-box overlays.
"""
from .page_images import save_page_images
from .pdfplumber_reader import extract_blocks

__all__ = ["extract_blocks", "save_page_images"]
