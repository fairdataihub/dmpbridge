"""pdfplumber-backed extractor (existing default)."""
from pathlib import Path

from .base import BaseExtractor


class PdfplumberExtractor(BaseExtractor):
    """Wrap the existing pdfplumber pipeline as a BaseExtractor.

    No additional dependencies — pdfplumber is already a core requirement.
    Produces full bbox and font metadata for every line-level block.

    Parameters
    ----------
    merge_lines:
        When ``True`` (default), wrapped lines are joined into paragraph-level
        blocks before classification, matching the granularity Docling and
        LightOnOCR produce natively.  Set ``False`` to get the raw line-level
        blocks — useful for reproducing earlier results, or for A/B comparison.
    """

    def __init__(self, merge_lines: bool = True) -> None:
        self.merge_lines = merge_lines

    def extract(self, pdf_path: Path) -> list[dict]:
        from ..preprocess import extract_blocks, merge_wrapped_lines
        blocks = extract_blocks(pdf_path)
        if self.merge_lines:
            blocks = merge_wrapped_lines(blocks)
        return blocks
