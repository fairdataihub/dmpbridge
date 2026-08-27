"""Docling-backed extractor — whole-document text with visual-signal markers.

Docling runs a layout model over each page and exports the document as
Markdown: section headings come out as ``## …`` lines, paragraphs as text,
bullets as ``- `` items, tables as pipe tables. That Markdown is then
translated into the same marker convention
:class:`~dmpbridge.extractors.pdfplumber_extractor.PdfplumberExtractor`
produces — headings wrapped in ``** … **``, italic in ``_…_`` — and returned
as one whole-document string, so
:class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy` classifies it with
the exact same system prompt, unchanged.

What Docling adds over pdfplumber, and what it does not
-------------------------------------------------------
Docling decides what is a heading from page *layout* (a trained layout
model), not from font metadata, so a heading set in the body font but on its
own line can still be recognised. It does **not** read bold/italic/underline
from the PDF: on this corpus ``item.formatting`` is ``None`` for every text
item (checked 2026-08-24 on samples 1, 3 and 6, docling 2.117), so the only
emphasis signal it contributes is the heading itself. In particular the
underlined headings in sample 6 are *not* detected — Docling returns that
document as one heading plus five list items.

Text coverage matches pdfplumber: the same probe found 0–1 words present in
one extractor's output and not the other's, per document. Reading *order*
can differ, though: sample 1's document title sits in the page-header
region, and Docling emits it after the rest of page 1 (line 45 of the
Markdown) rather than first. Nothing is lost, but a block the annotation
calls ``title`` arrives mid-document. Docling's own OCR
(RapidOCR) is enabled in auto mode, so it only fires on pages with no text
layer; on this corpus every page has one and conversion takes 0.1–3 s.

Install:
    pip install dmpbridge[docling]
    # or directly:
    pip install docling
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .base import BaseExtractor

# ``## Heading`` -> ``** Heading **``. The spaces inside the markers match
# what pdfplumber_reader emits for an emphasised line ("** Element 1: **"),
# so the classifier sees the same shape from both extractors.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")

# Docling's serializer writes italic as *single asterisks*; the project
# convention is _underscores_. Only single-asterisk runs are touched — a
# ``**bold**`` run is already in the shared convention and passes through.
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)")

# OCR / layout reconstruction spaces words inexactly, so runs of 2+ spaces
# show up mid-sentence ("Data  will  be  findable"). Collapsing horizontal
# whitespace only cannot touch the newlines that separate lines and blocks.
_EXTRA_SPACES = re.compile(r"[ \t]{2,}")
_EXTRA_BLANKS = re.compile(r"\n{3,}")


def markdown_to_marked_text(markdown: str) -> str:
    """Translate Docling Markdown into the pipeline's marker convention.

    Pure function, kept separate from the converter so it can be tested
    without Docling installed.
    """
    lines = []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            title = m.group(1).strip()
            lines.append(f"** {title} **" if title else "")
            continue
        lines.append(_ITALIC.sub(r"_\1_", line))
    text = "\n".join(lines)
    text = _EXTRA_SPACES.sub(" ", text)
    text = _EXTRA_BLANKS.sub("\n\n", text)
    return text.strip()


_NATIVE_PART_ORDER = ("version", "status", "timestamp", "timings", "errors",
                      "confidence", "document", "pages")


def native_result_dict(result) -> dict:
    """Docling's full native conversion result as one dict.

    ``ConversionResult.save()`` is Docling's own serialisation of everything it
    computed — document, per-page layout predictions, parsed page cells, page
    images, confidence — but it writes a zip of JSON parts. This merges the
    parts into a single dict, document first, pages last, so it can be written
    as one file. Requires the converter to have been run with
    ``generate_parsed_pages=True``; otherwise every ``parsed_page`` is null.
    """
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "native.zip"
        result.save(filename=bundle)
        with zipfile.ZipFile(bundle) as z:
            parts = {name[:-5]: json.loads(z.read(name).decode("utf-8"))
                     for name in z.namelist() if name.endswith(".json")}
    ordered = {k: parts[k] for k in _NATIVE_PART_ORDER if k in parts}
    ordered.update({k: v for k, v in parts.items() if k not in ordered})
    return ordered


class DoclingExtractor(BaseExtractor):
    """Convert a PDF with Docling and return it as one marked-up text block.

    Same one-item ``[{"text": ...}]`` return shape as the pdfplumber and
    LightOnOCR extractors, so stage 1 output is interchangeable between the
    three and the strategy and evaluation code need no change.

    OCR runs via Docling's auto mode (``do_ocr=True``) — pages with a usable
    text layer are read directly, OCR only fires on pages that need it.
    Table structure recovery is enabled so tables come through as pipe
    tables rather than as a jumble of cells.

    Parameters
    ----------
    save_native:
        Also write Docling's full native result as
        ``1_extracted/docling/<stem>.native.json`` — the parsed page cells
        (font name, size, position per line and word), hyperlinks, the layout
        model's clusters with confidences, and the page images. None of that
        survives into the Markdown; see
        ``notebooks/exploration-docling-native-format.ipynb``. Off by default:
        it makes Docling keep the parsed pages in memory and costs ~2–5 MB per
        document on disk. Does not change the extracted text.
    native_images:
        Include the page renders in the native file (most of its size). Only
        used when ``save_native`` is on.
    """

    def __init__(self, save_native: bool = False, native_images: bool = True) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed.\n"
                "Install with:  pip install dmpbridge[docling]\n"
                "          or:  pip install docling"
            ) from exc

        import logging
        logging.getLogger("docling").setLevel(logging.WARNING)
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        if save_native:
            # Docling discards the parsed pages (cells, fonts, links) once the
            # document is assembled unless told to keep them.
            pipeline_options.generate_parsed_pages = True
            pipeline_options.generate_page_images = native_images
        self._save_native = save_native

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # ── BaseExtractor protocol ────────────────────────────────────────────────

    def extract(self, pdf_path: Path) -> list[dict]:
        result = self._converter.convert(str(pdf_path))
        # escape_html=False: this text goes to an LLM prompt, not a browser,
        # so a literal ">" must stay ">" ("stored for > 10 years"), not
        # become "&gt;".
        markdown = result.document.export_to_markdown(escape_html=False)
        # Keep the untranslated Markdown next to the cached stage-1 JSON: when a
        # label looks wrong, the .md shows what Docling saw before translation.
        self._save_side_file(pdf_path, "md", markdown)
        if self._save_native:
            self._save_side_file(pdf_path, "native.json",
                                 json.dumps(native_result_dict(result), indent=2,
                                            ensure_ascii=False))
        return [{"text": markdown_to_marked_text(markdown)}]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _save_side_file(pdf_path: Path, suffix: str, content: str) -> None:
        """Write ``<stem>.<suffix>`` next to the cached stage-1 JSON. Best-effort:
        extraction must not fail because a side artifact could not be written."""
        try:
            from ..core.paths import EXTRACTED_DIR
            out = EXTRACTED_DIR / "docling" / f"{pdf_path.stem}.{suffix}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
        except OSError:
            pass

