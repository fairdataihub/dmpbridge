"""pdfplumber-backed extractor (existing default)."""
from pathlib import Path

from .base import BaseExtractor


class PdfplumberExtractor(BaseExtractor):
    """Wrap the existing pdfplumber pipeline as a BaseExtractor.

    No additional dependencies — pdfplumber is already a core requirement.
    Produces full bbox and font metadata for every line-level block.
    """

    def extract(self, pdf_path: Path) -> list[dict]:
        from ..preprocess import extract_blocks
        return extract_blocks(pdf_path)
