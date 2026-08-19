"""Extractor factory for the DMPBridge pipeline.

An extractor converts a PDF into a flat list of text block dicts.  All
extractors implement :class:`BaseExtractor` and produce the same schema so
the downstream strategy and evaluation code are unaffected by the choice.

Supported extractors
--------------------
``"pdfplumber"``  — whole-document text with **bold**/_italic_/++underline++
                   visual-signal markers, extraction and labeling fused into
                   one call. Font/size-based (bold, italic) and drawn-shape-based
                   (underline) signal detection — no vision model, CPU only.
``"lightonocr"``  — same whole-document, same marker convention, but read by
                   an actual vision-LLM (LightOnOCR-2-1B) instead of parsed
                   from PDF font/shape metadata. Requires a CUDA GPU and the
                   ``dmpbridge[lighton]`` extras (torch, transformers, pymupdf).
                   Confirmed (2026-08-19) to score lower overall than pdfplumber
                   on this project's corpus (81.3% vs 90.7% pooled F1) and ~14x
                   slower — kept as an available alternative extractor, not the
                   default, since it occasionally does better on a specific
                   document (see notebooks/comparison-gemma-pdfplumber-vs-lightonocr.ipynb).

Usage
-----
    from dmpbridge.extractors import get_extractor

    extractor = get_extractor("pdfplumber")
    blocks    = extractor.extract(Path("document.pdf"))
"""
from .base import BaseExtractor

_EXTRACTORS = {
    "pdfplumber": ("pdfplumber_extractor", "PdfplumberExtractor"),
    "lightonocr": ("lighton_extractor",    "LightOnExtractor"),
}


def get_extractor(name: str, **kwargs) -> BaseExtractor:
    """Return a configured extractor instance by name.

    Parameters
    ----------
    name:
        ``"pdfplumber"`` or ``"lightonocr"``.
    **kwargs:
        Passed through to the extractor constructor.
    """
    if name not in _EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {name!r}. "
            f"Supported values: {', '.join(repr(k) for k in _EXTRACTORS)}."
        )
    module_name, class_name = _EXTRACTORS[name]
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, class_name)(**kwargs)


__all__ = ["BaseExtractor", "get_extractor"]
