"""Abstract base class for all PDF extractors."""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseExtractor(ABC):
    """Extract text from a PDF file, returned as a flat list of block dicts.

    Every extractor implemented so far (pdfplumber, LightOnOCR, Docling)
    returns the whole document as a single ``[{"text": "..."}]`` entry, with
    **bold**/_italic_/++underline++ visual-signal markers embedded in the
    text rather than as separate fields; see
    :class:`~dmpbridge.extractors.pdfplumber_extractor.PdfplumberExtractor`.
    There is no shared multi-block schema to conform to beyond that — a
    future extractor is free to define its own block shape, as long as
    :meth:`~dmpbridge.strategies.wholedoc.WholeDocStrategy.classify_entire_document`
    (or a new classification path built alongside it) is written to match.
    """

    @abstractmethod
    def extract(self, pdf_path: Path) -> list[dict]:
        """Return a flat list of block dicts from *pdf_path*."""
        ...
