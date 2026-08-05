"""Extractor factory for the DMPBridge pipeline.

An extractor converts a PDF into a flat list of text block dicts.  All
extractors implement :class:`BaseExtractor` and produce the same schema so
the downstream strategy and evaluation code are unaffected by the choice.

Supported extractors
--------------------
``"pdfplumber"`` — line-level extraction, full bbox + font metadata (default).
                  Wrapped lines are merged into paragraphs unless
                  ``merge_lines=False`` is passed.
``"docling"``    — ML layout analysis, understands headings/tables/lists
``"lighton"``    — LightOnOCR-2-1B vision-LLM, handles scanned / complex PDFs

Usage
-----
    from dmpbridge.extractors import get_extractor

    extractor = get_extractor("docling")
    blocks    = extractor.extract(Path("document.pdf"))
"""
from .base import BaseExtractor


def get_extractor(name: str, **kwargs) -> BaseExtractor:
    """Return a configured extractor instance by name.

    Parameters
    ----------
    name:
        ``"pdfplumber"`` | ``"docling"`` | ``"lighton"``
    **kwargs:
        Passed through to the extractor constructor.
        Recognised by ``"pdfplumber"``: ``merge_lines``.
        Recognised by ``"lighton"``: ``model_id``, ``max_new_tokens``.
    """
    if name == "pdfplumber":
        from .pdfplumber_extractor import PdfplumberExtractor
        return PdfplumberExtractor(**kwargs)

    if name == "docling":
        from .docling_extractor import DoclingExtractor
        return DoclingExtractor()

    if name == "lighton":
        from .lighton_extractor import LightOnExtractor
        return LightOnExtractor(**kwargs)

    raise ValueError(
        f"Unknown extractor {name!r}. "
        "Supported values: 'pdfplumber', 'docling', 'lighton'."
    )


__all__ = ["BaseExtractor", "get_extractor"]
