"""pdfplumber-backed extractor — whole-document text with visual-signal markers."""
from pathlib import Path

from .base import BaseExtractor


class PdfplumberExtractor(BaseExtractor):
    """Wrap pdfplumber's font-baseline visual-signal extraction as a BaseExtractor.

    No additional dependencies — pdfplumber is already a core requirement.

    Unlike Docling and LightOnOCR, this does not segment the document into
    blocks at extraction time: it returns the whole document as a single
    text string (wrapped in a one-item list so the return shape still
    matches :class:`BaseExtractor`), with words visually emphasized relative
    to the document's own body-text baseline wrapped in ``**...**`` and
    italic words in ``_..._``. :class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy`
    classifies this text and splits it into labeled entries in one model
    call — pdfplumber is the only extractor where extraction and labeling
    are not separate steps.
    """

    def extract(self, pdf_path: Path) -> list[dict]:
        from ..preprocess import extract_text_for_llm
        return [{"text": extract_text_for_llm(pdf_path)}]
