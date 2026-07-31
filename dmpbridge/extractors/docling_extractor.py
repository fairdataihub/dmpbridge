"""Docling-backed extractor — ML layout analysis for PDF understanding.

Docling (https://github.com/DS4SD/docling) uses deep-learning layout models
to identify headings, paragraphs, tables, lists, and captions — giving
richer structure than rule-based pdfplumber extraction.

Install:
    pip install dmpbridge[docling]
    # or directly:
    pip install docling
"""
from pathlib import Path

from .base import BaseExtractor

# DocItemLabel values that we treat as visually bold / heading blocks.
_HEADING_LABEL_NAMES = {"title", "section_header", "page_header"}


class DoclingExtractor(BaseExtractor):
    """Convert a PDF to blocks using Docling's DocumentConverter.

    The Docling document is flattened to the same block schema used by
    pdfplumber so that the downstream WholedocStrategy is unaffected.

    Bold detection is derived from the item label (TITLE / SECTION_HEADER)
    rather than font inspection, which is unavailable after ML extraction.
    """

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import (
                PipelineOptions,
                AcceleratorOptions,
                AcceleratorDevice,
            )
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed.\n"
                "Install it with:  pip install dmpbridge[docling]\n"
                "            or:  pip install docling"
            ) from exc

        # Suppress noisy Docling startup logging
        import logging
        logging.getLogger("docling").setLevel(logging.WARNING)

        pipeline_options = PipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.AUTO,
        )
        self._converter = DocumentConverter(pipeline_options=pipeline_options)

    # ── BaseExtractor protocol ────────────────────────────────────────────────

    def extract(self, pdf_path: Path) -> list[dict]:
        result = self._converter.convert(str(pdf_path))
        return self._document_to_blocks(result.document)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _document_to_blocks(self, doc) -> list[dict]:
        blocks: list[dict] = []

        for item, _level in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text or not text.strip():
                continue

            prov = item.prov[0] if getattr(item, "prov", None) else None
            page_no = prov.page_no if prov else 1
            bbox    = prov.bbox   if prov else None

            label_name = (
                item.label.name.lower()
                if hasattr(getattr(item, "label", None), "name")
                else ""
            )
            is_heading = label_name in _HEADING_LABEL_NAMES

            blocks.append({
                "page":          page_no,
                "line_order":    len(blocks),
                "text":          text.strip(),
                "x0":            float(bbox.l) if bbox else None,
                "top":           float(bbox.t) if bbox else None,
                "x1":            float(bbox.r) if bbox else None,
                "bottom":        float(bbox.b) if bbox else None,
                "avg_font_size": None,
                "font_names":    [],
                "is_bold":       is_heading,
                "is_italic":     False,
                "label":         None,
            })

        return blocks
