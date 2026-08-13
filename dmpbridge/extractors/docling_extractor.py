"""Docling-backed extractor — OCR, full Markdown export, section-by-section.

Adapted from a standalone ``test_docling.py`` script: OCR is enabled so
scanned/image-only PDFs still produce text, the whole document is exported
to Markdown (headings, tables, and lists preserved), and that Markdown is
split into sections at each heading line rather than read line by line.

**Known, accepted limitation — read before debugging a bad label.** A
section becomes exactly two blocks: the heading, and *everything* under it
up to the next heading, joined into one body block. If a DMP section
contains both a question and its answer, both end up in that single body
block, which can only be given one label. This is not a bug to fix here —
it is the deliberate trade-off of grouping by section rather than by
paragraph/list item, chosen in preference to finer-grained splitting.

**Page numbers are not available.** The Markdown export flattens the whole
document into one string with no per-line page marker, so unlike the
pdfplumber extractor this one cannot report which page a block came from —
every block is recorded as page 1, which is a placeholder, not a
measurement.

Install:
    pip install dmpbridge[docling]
    # or directly:
    pip install docling
"""
import re
from pathlib import Path

from .base import BaseExtractor

# OCR reconstructs word spacing inexactly, so runs of 2+ spaces/tabs show up
# mid-sentence ("Types  of  data") where the source had one. Collapsing them
# only touches horizontal whitespace of length 2+, so it cannot affect the
# single newlines this module relies on to keep numbered items on separate
# lines, or the single space after a markdown "#"/"-" marker.
_EXTRA_SPACES = re.compile(r"[ \t]{2,}")


def _clean_text(text: str) -> str:
    return _EXTRA_SPACES.sub(" ", text)


def _split_into_sections(markdown: str) -> list[tuple[str, str, str]]:
    """Group Markdown into (heading_markdown, heading_clean, body) triples.

    A new section starts at every line beginning with ``#``. Everything
    between one heading and the next — however many paragraphs, list items,
    or table rows — is joined into that section's single body string.

    ``heading_markdown`` keeps the ``#``/``##`` prefix verbatim, so a block
    built from it reads as actual markdown rather than a bare label —
    stripping it out (as an earlier version of this function did) meant the
    output looked nothing like the ``.md`` file it was exported from.
    ``heading_clean`` has the prefix removed and exists only for the
    ``section`` grouping key, which is meant to be a readable name, not
    markdown.
    """
    sections: list[tuple[str, str, str]] = []
    heading_md, heading_clean = "(preamble)", "(preamble)"
    lines: list[str] = []

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if lines or heading_clean != "(preamble)":
                body = _clean_text("\n".join(lines).strip())
                sections.append((heading_md, heading_clean, body))
            heading_md = _clean_text(stripped)
            heading_clean = _clean_text(stripped.lstrip("#").strip()) or "(untitled)"
            lines = []
        else:
            lines.append(line)

    if lines or heading_clean != "(preamble)":
        body = _clean_text("\n".join(lines).strip())
        sections.append((heading_md, heading_clean, body))
    return sections


class DoclingExtractor(BaseExtractor):
    """Convert a PDF to one heading block + one body block per section.

    OCR runs via Docling's default auto mode (``do_ocr=True``) — pages with
    a usable text layer are read directly, OCR only fires on pages that
    need it. Table structure recovery is enabled.
    """

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed.\n"
                "Install it with:  pip install docling"
            ) from exc

        import logging
        logging.getLogger("docling").setLevel(logging.WARNING)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # ── BaseExtractor protocol ────────────────────────────────────────────────

    def extract(self, pdf_path: Path) -> list[dict]:
        result = self._converter.convert(str(pdf_path))
        # escape_html=False: this text goes to an LLM prompt, not a browser,
        # so there is no reason to turn a literal ">" into "&gt;" — the
        # default left that unconverted in every block ("stored for &gt; 10
        # years" instead of "> 10 years").
        markdown = result.document.export_to_markdown(escape_html=False)
        self._save_markdown(pdf_path, markdown)
        return self._sections_to_blocks(_split_into_sections(markdown))

    @staticmethod
    def _save_markdown(pdf_path: Path, markdown: str) -> None:
        """Write the whole-document Markdown next to the cached block JSON,
        so it can be opened and read as an actual ``.md`` file rather than
        only reconstructed from block text. Best-effort: extraction must not
        fail just because this side artifact could not be written.
        """
        try:
            from dmpbridge.core.paths import EXTRACTED_DIR
            out = EXTRACTED_DIR / "docling" / f"{pdf_path.stem}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown, encoding="utf-8")
        except OSError:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sections_to_blocks(self, sections: list[tuple[str, str, str]]) -> list[dict]:
        blocks: list[dict] = []
        for heading_md, heading_clean, body in sections:
            if heading_clean != "(preamble)":
                self._append(blocks, heading_md, is_heading=True, section=heading_clean)
            if body:
                self._append(blocks, body, is_heading=False, section=heading_clean)
        return blocks

    @staticmethod
    def _append(blocks: list[dict], text: str, *, is_heading: bool,
                section: str) -> None:
        # No x0/top/x1/bottom/avg_font_size/font_names: this extractor works
        # from Docling's flattened whole-document markdown export, which has
        # no per-block geometry or font data to report — putting null/[] in
        # every block for fields that can never be populated just added
        # noise. Confirmed safe to omit: page_images.py already guards bbox
        # access with `b.get("x0") is None`, which treats a missing key the
        # same as a null one, and nothing else in the pipeline reads these
        # fields at all. `page` and `is_bold` stay — both are read directly
        # (`b["page"]`, `b["is_bold"]`) elsewhere and would break if absent,
        # even though `page` here is a placeholder (always 1), not a
        # measurement.
        blocks.append({
            "page":          1,
            "line_order":    len(blocks),
            "text":          text,
            "section":       section,
            "is_bold":       is_heading,
            "is_italic":     False,
        })
