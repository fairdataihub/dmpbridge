"""Abstract base class for all PDF extractors."""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseExtractor(ABC):
    """Extract text blocks from a PDF file.

    All concrete extractors must produce a flat list of block dicts that
    conform to the shared schema expected by WholeDocStrategy:

        text          str   — cleaned text of the block
        page          int   — 1-based page number
        is_bold       bool  — True when the block is visually bold / a heading
        is_italic     bool  — True when italic (False if OCR-only)
        line_order    int   — position within the document (for ordering)
        x0/top/x1/bottom  float | None  — bounding box (None for OCR extractors)
        avg_font_size     float | None  — mean character size (None for OCR)
        font_names        list[str]     — font names seen ([] for OCR)
    """

    @abstractmethod
    def extract(self, pdf_path: Path) -> list[dict]:
        """Return a flat list of block dicts from *pdf_path*."""
        ...
