"""Docling-backed extractor — section-aware, OCR, markdown-structured.

Walks Docling's own item tree (``iterate_items()``) rather than its flattened
markdown string, so each block keeps real page/bbox provenance *and* carries
markdown heading/list syntax (``#``/``##``/``- ``) directly in its text — a
stronger signal to the labeling model than a bare boolean flag.

Heading detection is Docling's layout model alone (``TITLE`` /
``SECTION_HEADER`` labels) — no lexical pattern-matching layered on top. An
earlier version added regex rules to split "run-in" headings (a heading
bolded but typed on the same line as its paragraph, e.g. "1. Types of data.
The bulk of the data generated..."), which Docling's layout model emits as a
single TEXT item. Those rules worked and were checked for false positives,
but only against this project's fixed 10-document corpus — the same data
they were designed against, not an independent test of whether they
generalize to a DMP nobody has looked at yet. Removed rather than kept on
the strength of "passed the checks I could run": a rule tuned and validated
on the same small set it will also be applied to is not evidence it holds up
elsewhere. If that gap needs closing again, it should be with an independent
document to test against, not more pattern-tuning on these 10.

The practical effect: a heading typed on the same line as its own paragraph
stays one block, one label. Documents where that happens (known here: sample
6, and part of sample 2) will still get a lower score than documents without
that layout — a real, disclosed limitation of the source PDF, not something
this extractor works around anymore.

Every block carries a ``section`` field — the most recent heading Docling
itself detected — so blocks can be grouped by section without collapsing
them into one large per-section block. Coarser, section-level chunking was
considered and rejected: this pipeline labels one block at a time, and a
block spanning two DMP fields (e.g. a description and the answer that
follows it) cannot be given two labels.

Install:
    pip install dmpbridge[docling]
    # or directly:
    pip install docling
"""
from pathlib import Path

from .base import BaseExtractor

_SKIP_LABELS = {"page_header", "page_footer"}
_HEADING_LABELS = {"title", "section_header"}
_HEADING_HASHES = {"title": "#", "section_header": "##"}


class DoclingExtractor(BaseExtractor):
    """Convert a PDF to blocks using Docling's own layout model, OCR, and
    its table structure recognition — no lexical rules on top.

    OCR runs on every page (``force_full_page_ocr=True``) rather than only on
    pages without a text layer — treated as a robustness setting to test
    deliberately, not a scanned-page fallback.
    """

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                AcceleratorOptions,
                AcceleratorDevice,
                OcrAutoOptions,
            )
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed.\n"
                "Install it with:  pip install docling"
            ) from exc

        import logging
        logging.getLogger("docling").setLevel(logging.WARNING)

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=OcrAutoOptions(force_full_page_ocr=True),
            do_table_structure=True,
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.AUTO,
                num_threads=4,
            ),
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # ── BaseExtractor protocol ────────────────────────────────────────────────

    def extract(self, pdf_path: Path) -> list[dict]:
        result = self._converter.convert(str(pdf_path))
        return self._document_to_blocks(result.document)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _document_to_blocks(self, doc) -> list[dict]:
        from docling_core.types.doc.document import TableItem

        blocks: list[dict] = []
        section = "(preamble)"

        for item, _level in doc.iterate_items():
            label = getattr(getattr(item, "label", None), "name", "").lower()
            if label in _SKIP_LABELS:
                continue

            if isinstance(item, TableItem):
                text = item.export_to_markdown(doc=doc).strip()
                if not text:
                    continue
                self._append(blocks, item, text, is_heading=False, section=section)
                continue

            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue

            if label in _HEADING_LABELS:
                section = text
                hashes = _HEADING_HASHES.get(label, "##")
                self._append(blocks, item, f"{hashes} {text}", is_heading=True,
                             section=section)
                continue

            prefix = "- " if label == "list_item" else ""
            self._append(blocks, item, f"{prefix}{text}", is_heading=False,
                         section=section)

        return blocks

    @staticmethod
    def _append(blocks: list[dict], item, text: str, *, is_heading: bool,
                section: str | None = None) -> None:
        prov = item.prov[0] if getattr(item, "prov", None) else None
        bbox = prov.bbox if prov else None
        blocks.append({
            "page":          prov.page_no if prov else 1,
            "line_order":    len(blocks),
            "text":          text,
            "section":       section,
            "x0":            float(bbox.l) if bbox else None,
            "top":           float(bbox.t) if bbox else None,
            "x1":            float(bbox.r) if bbox else None,
            "bottom":        float(bbox.b) if bbox else None,
            "avg_font_size": None,
            "font_names":    [],
            "is_bold":       is_heading,
            "is_italic":     False,
        })
