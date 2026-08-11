"""pdfplumber-backed extractor (existing default)."""
from pathlib import Path

from .base import BaseExtractor


class PdfplumberExtractor(BaseExtractor):
    """Wrap the existing pdfplumber pipeline as a BaseExtractor.

    No additional dependencies — pdfplumber is already a core requirement.
    Produces line-level blocks with full bbox and font metadata.

    Note that pdfplumber segments by *line*, so a wrapped paragraph arrives as
    several blocks, unlike Docling and LightOnOCR which segment by paragraph.
    """

    def extract(self, pdf_path: Path) -> list[dict]:
        from ..preprocess import extract_blocks
        return extract_blocks(pdf_path)
