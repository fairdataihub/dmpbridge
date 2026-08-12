"""Docling-backed extractor — OCR + markdown-structured PDF understanding.

Docling (https://github.com/DS4SD/docling) runs full-page OCR and a
deep-learning layout model, then exports each page as markdown: headings
carry ``#``/``##`` prefixes, list items carry ``- ``. Those markers are kept
verbatim in block text — a stronger, human-readable heading signal than a
bare ``is_bold`` flag, visible directly to the labeling model.

Install:
    pip install dmpbridge[docling]
    # or directly:
    pip install docling
"""
import re
from pathlib import Path

from .base import BaseExtractor

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_IMAGE_PLACEHOLDER = "<!-- image -->"


class DoclingExtractor(BaseExtractor):
    """Convert a PDF to blocks via Docling's markdown export.

    OCR runs on every page (``force_full_page_ocr=True``) rather than only on
    pages without a text layer — this project treats OCR as a robustness
    setting to test deliberately, not a scanned-page fallback.

    Each page's markdown is split first on blank lines (its paragraph/element
    boundary), then each resulting piece is split again on single newlines.
    The second split matters: Docling sometimes serialises a run of list
    items (e.g. a numbered set of section headings) as one blank-line-
    delimited chunk with the items joined by single ``\\n`` rather than each
    getting its own bullet — verified on sample 6, where five numbered
    headings would otherwise collapse into one block. Plain wrapped prose has
    no internal single-newline breaks in Docling's export, so the second
    split does not fragment ordinary paragraphs.
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
        blocks: list[dict] = []

        for page_no in sorted(doc.pages.keys()):
            md = doc.export_to_markdown(
                page_no=page_no, escape_html=False, escape_underscores=False,
            )
            for chunk in md.split("\n\n"):
                for line in chunk.split("\n"):
                    line = line.strip()
                    if not line or line == _IMAGE_PLACEHOLDER:
                        continue

                    # The markdown marker stays in `text` itself — not stripped —
                    # so the labeling model sees "## Heading" directly rather than
                    # relying only on the separate `is_bold` field to know it.
                    heading = _HEADING_RE.match(line)
                    is_heading = bool(heading)
                    text = line

                    blocks.append({
                        "page":          page_no,
                        "line_order":    len(blocks),
                        "text":          text,
                        "x0":            None,
                        "top":           None,
                        "x1":            None,
                        "bottom":        None,
                        "avg_font_size": None,
                        "font_names":    [],
                        "is_bold":       is_heading,
                        "is_italic":     False,
                    })

        return blocks
