"""Extractor factory for the DMPBridge pipeline.

An extractor converts a PDF into a flat list of text block dicts.  All
extractors implement :class:`BaseExtractor` and produce the same schema so
the downstream strategy and evaluation code are unaffected by the choice.

Supported extractors
--------------------
``"pdfplumber"`` — whole-document text with **bold**/_italic_ visual-signal
                  markers, extraction and labeling fused into one call (default).
``"lighton"``    — LightOnOCR-2-1B vision-LLM, handles scanned / complex PDFs

Usage
-----
    from dmpbridge.extractors import get_extractor

    extractor = get_extractor("lighton")
    blocks    = extractor.extract(Path("document.pdf"))
"""
from .base import BaseExtractor


def get_extractor(name: str, **kwargs) -> BaseExtractor:
    """Return a configured extractor instance by name.

    Parameters
    ----------
    name:
        ``"pdfplumber"`` | ``"lighton"``
    **kwargs:
        Passed through to the extractor constructor.
        Recognised by ``"lighton"``: ``model_id``, ``max_new_tokens``.
    """
    if name == "pdfplumber":
        from .pdfplumber_extractor import PdfplumberExtractor
        return PdfplumberExtractor(**kwargs)

    if name == "lighton":
        from .lighton_extractor import LightOnExtractor
        return LightOnExtractor(**kwargs)

    raise ValueError(
        f"Unknown extractor {name!r}. "
        "Supported values: 'pdfplumber', 'lighton'."
    )


__all__ = ["BaseExtractor", "get_extractor"]
