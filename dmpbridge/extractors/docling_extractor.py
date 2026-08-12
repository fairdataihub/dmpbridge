"""Docling-backed extractor — section-aware, OCR, markdown-structured.

Walks Docling's own item tree (``iterate_items()``) rather than its flattened
markdown string, so each block keeps real page/bbox provenance *and* carries
markdown heading/list syntax (``#``/``##``/``- ``) directly in its text — a
stronger signal to the labeling model than a bare boolean flag.

Docling's layout model is the primary heading signal (``TITLE`` /
``SECTION_HEADER`` labels). It is not the only one: DMP section headings are
often "run-in" — bolded but typed on the same line as the paragraph they
introduce ("1. Types of data. The bulk of the data generated...", or with no
number at all: "Roles & Responsibilities. For the proposed research...").
Docling's layout model emits that as a single TEXT item, so a lexical
fallback (below) splits it into a heading block and a body block. Both
regexes were validated against every block already extracted across this
project's 10 documents, checking for false positives, not assumed correct —
see ``_NUMBERED_RUN_IN`` and ``_BARE_RUN_IN`` for what was found.

Every block also carries a ``section`` field — the heading text most
recently seen — so blocks can be grouped by section without collapsing them
into one large per-section block. Coarser, section-level chunking was
considered and rejected: this pipeline labels one block at a time, and a
block spanning two DMP fields (e.g. a description and the answer that
follows it) cannot be given two labels.

Install:
    pip install dmpbridge[docling]
    # or directly:
    pip install docling
"""
import re
from pathlib import Path

from .base import BaseExtractor

_SKIP_LABELS = {"page_header", "page_footer"}
_HEADING_LABELS = {"title", "section_header"}
_HEADING_HASHES = {"title": "#", "section_header": "##"}

# A heading given its own line, with no body run into it — "1. Types of data"
# or "IV. Roles and Responsibilities" with nothing else on the line.
_NUMBERED_STANDALONE = re.compile(
    r"""^\s*(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])\s*[.\):]\s+
        [A-Z(\"'“].{2,110}$""",
    re.VERBOSE,
)

# A numbered/lettered heading run into its own body on one line — verified
# against all 10 documents' extracted output: matches only the 5 genuine
# fused headings in sample 6 (each "N. Heading. Body..."), zero elsewhere.
_NUMBERED_RUN_IN = re.compile(
    r"""^\s*
    (?P<title>(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])\s*[.\):]\s+[A-Z][^.]{2,110}?)
    \s*[.:]\s+
    (?P<body>[A-Z"'“(].+)$""",
    re.VERBOSE | re.DOTALL,
)

# A bare heading phrase (no number) run into its body — "Roles &
# Responsibilities. For the proposed research..." Every real DMP heading in
# this corpus is a Title-Case phrase of 2+ words; requiring that excludes
# abbreviations like "Prof. Leahey's work..." (a false positive at 1 word,
# fixed by requiring at least 2). One accepted false-positive risk remains —
# "Produced Data: Datasets will be..." in sample 5 also matches this
# pattern and may not be a real heading; splitting it is a safe default
# since the labeling model still assigns the final label either way.
_BARE_RUN_IN = re.compile(
    r"""^\s*
    (?P<title>[A-Z][\w']*(?:[ \t]+(?:&|and|of|for|in|to|the|or)?[ \t]*
               [A-Z][\w']*){1,7})
    \s*[.:]\s+
    (?P<body>[A-Z"'“(].{20,})$""",
    re.VERBOSE | re.DOTALL,
)

# List items are Docling's other run-in-heading source, and need a separate,
# looser pattern: Docling strips a list item's own "1."/"IV." marker out of
# ``item.text`` (its markdown renderer re-adds it later), so the number this
# extractor would otherwise require is simply not there to match on — and
# these particular headings turn out to be plain sentence case, not Title
# Case, so `_BARE_RUN_IN` above does not catch them either. The only signal
# left is structural: Docling already told us this item is an enumerated
# list entry, which is itself real evidence, so the per-word-capitalisation
# check is dropped for this pattern alone. Checked against every list item
# extracted across all 10 documents: matches exactly the 5 genuine fused
# headings in sample 6 and none of the other 15 (ordinary list entries start
# lowercase, mid-sentence, or have no second sentence to split off at all).
_LIST_ITEM_RUN_IN = re.compile(
    r"""^\s*
    (?P<title>[A-Z][^.]{2,90}?)
    \s*[.:]\s+
    (?P<body>[A-Z"'“(].{20,})$""",
    re.VERBOSE | re.DOTALL,
)


class DoclingExtractor(BaseExtractor):
    """Convert a PDF to blocks using Docling's layout model, OCR, and a
    lexical run-in-heading fallback.

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
                self._append(blocks, item, text, is_heading=False)
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

            # Layout model called this body text — check whether it is
            # really a heading (standalone or run into its own body) before
            # accepting that.
            if _NUMBERED_STANDALONE.match(text):
                section = text
                self._append(blocks, item, f"## {text}", is_heading=True,
                             section=section)
                continue

            if label == "list_item":
                run_in = _LIST_ITEM_RUN_IN.match(text)
            else:
                run_in = _NUMBERED_RUN_IN.match(text) or _BARE_RUN_IN.match(text)

            if run_in:
                section = run_in.group("title").strip()
                self._append(blocks, item, f"## {section}", is_heading=True,
                             section=section)
                self._append(blocks, item, run_in.group("body").strip(),
                             is_heading=False, section=section)
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
