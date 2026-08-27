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
``"docling"``     — same whole-document, same marker convention, built from
                   Docling's *native* page cells: pdfplumber's bold/italic
                   rules on each word's font name and size, hyperlink
                   rectangles as underline, and Docling's own heading label
                   where the font marks nothing. Markers match pdfplumber's
                   on 8 of 10 documents; 0.924 F1 vs pdfplumber's 0.946 with
                   gemma4:e4b, the gap being sample 6's drawn underlines,
                   which Docling has no shape data for. CPU-capable, 0.1–3 s
                   per document, requires the ``dmpbridge[docling]`` extra.

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
    "docling":    ("docling_extractor",    "DoclingExtractor"),
}


def get_extractor(name: str, **kwargs) -> BaseExtractor:
    """Return a configured extractor instance by name.

    Parameters
    ----------
    name:
        ``"pdfplumber"``, ``"lightonocr"`` or ``"docling"``.
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
