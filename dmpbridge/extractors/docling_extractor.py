"""Docling-backed extractor — whole-document text with visual-signal markers,
built from Docling's *native* page cells.

Docling runs a layout model over each page and assembles a document of typed
blocks. Its Markdown and JSON exports keep only the blocks' text and labels;
the fonts, sizes and hyperlinks it read from the PDF are discarded once the
document is assembled — Docling's PDF pipeline never sets ``formatting`` on a
block. This extractor keeps the parsed pages (``generate_parsed_pages=True``)
and builds the annotated blob from them with the same rules
:mod:`~dmpbridge.preprocess.pdfplumber_reader` applies to pdfplumber's
characters — bold from the font name, size within the body face, italic from
the font name, underline from hyperlink rectangles — plus Docling's own
heading label where the font marks nothing. See :func:`native_marked_text`.
The result uses the same ``**bold**`` / ``_italic_`` / ``++underline++``
convention as :class:`~dmpbridge.extractors.pdfplumber_extractor.PdfplumberExtractor`,
so :class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy` classifies it
with the exact same system prompt, unchanged.

Measured 2026-08-27 (gemma4:e4b, samples 1–10): markers identical to
pdfplumber's on 8 of 10 documents; F1 0.924 Path A / 0.910 Path B against
pdfplumber's 0.946 / 0.951 — ahead on samples 2 and 5, level on seven, behind
only on sample 6. The earlier Markdown-based version of this extractor
(``markdown_to_marked_text``, kept here as ``source="markdown"``) scored 0.767.

What Docling cannot give
------------------------
A drawn underline with no hyperlink behind it. Docling's parsed page reports
hyperlink rectangles but no drawn shapes, so sample 6's underlined headings
are invisible at every level (pdfplumber reads them from the page's
rectangles). Docling's own OCR (RapidOCR) is enabled in auto mode, so it only
fires on pages with no text layer; on this corpus every page has one and
conversion takes 0.1–3 s.

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


# ── Native source: build the marked text from Docling's page cells ─────────────
#
# The same three rules pdfplumber_reader applies to pdfplumber's characters,
# applied to Docling's word cells, which carry the same font names and sizes:
#   emphasised  = font differs from the body font and is not an italic variant,
#                 or the word is set larger than the body size (+0.5 pt)
#   italic      = font name says Italic / Oblique
#   underlined  = the word sits inside a hyperlink rectangle (Docling reports
#                 links; it has no drawn-shape data, so a plain drawn underline
#                 is not recoverable) — URLs excluded, as pdfplumber excludes them
# Plus one signal pdfplumber cannot give: a Docling ``section_header`` block is
# emphasised as a whole, even when its font does not say so.

_BOLD_TAGS = ("Bold", "Black", "Heavy", "Semibold", "bold", "black", "heavy", "semibold")
_ITALIC_TAGS = ("Italic", "Oblique", "italic", "oblique")
_URL_TAGS = ("http:", "https:", "www.")


def _looks_like_url(text: str) -> bool:
    lower = text.lower()
    return any(tag in lower for tag in _URL_TAGS)


def _word_flags(text: str, font: str, height: float, body_font: str, body_size: float,
                in_link: bool) -> tuple[bool, bool, bool]:
    """(emphasised, italic, underlined) for one word — pdfplumber's rules.

    One adaptation: pdfplumber compares font *sizes*; Docling's cells only give
    the glyph rectangle's height, which differs between faces at the same size
    (sample 10: BoldItalic headings measure 12.2 against a 9.0 body, all 11 pt in
    the PDF). So the size test applies only within the body face — where a
    taller rectangle really is a larger size (samples 3 and 9 set headings in
    the body face at 12.6–13.2 against 11.4).
    """
    italic = any(tag in font for tag in _ITALIC_TAGS)
    bigger = font == body_font and height > body_size + 0.5
    if any(ch.isalnum() for ch in text):
        emph = (font != body_font and not italic) or bigger
    else:
        # punctuation: a font swap alone is not bold evidence (see pdfplumber_reader)
        emph = any(tag in font for tag in _BOLD_TAGS) or bigger
    under = in_link and not _looks_like_url(text)
    return emph, italic, under


def _render_line(words: list[tuple[str, bool, bool, bool]]) -> str:
    """Join (text, emph, italic, under) words into one line with **, _, ++ markers,
    opening and closing runs exactly as pdfplumber_reader._extract_page_lines does."""
    rendered: list[str] = []
    open_bold = open_italic = open_under = False
    for text, emph, ital, under in words:
        if open_under and not under:
            rendered.append("++"); open_under = False
        if open_italic and not ital:
            rendered.append("_"); open_italic = False
        if open_bold and not emph:
            rendered.append("**"); open_bold = False
        if emph and not open_bold:
            rendered.append("**"); open_bold = True
        if ital and not open_italic:
            rendered.append("_"); open_italic = True
        if under and not open_under:
            rendered.append("++"); open_under = True
        rendered.append(text)
    if open_under:
        rendered.append("++")
    if open_italic:
        rendered.append("_")
    if open_bold:
        rendered.append("**")
    line = " ".join(rendered)
    return line.replace("** **", " ").replace("_ _", " ").replace("++ ++", " ")


def native_marked_text(result) -> str:
    """Build the annotated text blob from a Docling ConversionResult's native data.

    Takes the document's blocks (body layer only) in page order, top to bottom
    — not Docling's reading order, which places a title set in the page-header
    region *after* the first section (samples 1 and 8); pdfplumber reads top
    to bottom, and these documents are single-column. For each block takes the
    word cells inside its box on the page, groups them into lines, flags each
    word from its font, size and hyperlink membership, and renders the lines
    with the project's markers. Lines and blocks are joined with single
    newlines, as pdfplumber's blob is — blank lines between blocks were tried
    first and cost points on samples 5 and 8. A Docling ``section_header``
    is emphasised as a whole only when its font marks nothing; a heading the
    font already sets italic stays ``_…_`` like pdfplumber's. Requires the
    converter to have been run with ``generate_parsed_pages=True``.
    """
    from collections import Counter

    pages = {p.page_no: p for p in result.pages}

    def units(p):
        """Word cells, or line cells where OCR produced no word level.

        Full-page OCR (force_ocr) fills ``textline_cells`` but leaves
        ``word_cells`` empty; without this fallback the whole text came back
        empty and the model hallucinated a document (seen on sample 11).
        """
        return p.parsed_page.word_cells or p.parsed_page.textline_cells

    # body font profile, weighted by characters like pdfplumber's
    sizes: Counter = Counter()
    fonts: Counter = Counter()
    for p in pages.values():
        for w in units(p):
            n = len(w.text)
            sizes[round(w.rect.height, 1)] += n
            # OCR cells are plain TextCells with no font attribute
            fonts[getattr(w, "font_name", "")] += n
    if not sizes:
        return ""
    body_size = sizes.most_common(1)[0][0]
    body_font = fonts.most_common(1)[0][0]

    # per page: words with top-left boxes, and the hyperlink rectangles
    page_words: dict[int, list] = {}
    page_links: dict[int, list] = {}
    for no, p in pages.items():
        H = p.size.height
        page_words[no] = [(w, w.rect.to_bounding_box().to_top_left_origin(page_height=H))
                          for w in units(p)]
        page_links[no] = [h.rect.to_bounding_box().to_top_left_origin(page_height=H)
                          for h in p.parsed_page.hyperlinks]

    def in_link(b, links) -> bool:
        cx, cy = (b.l + b.r) / 2, (b.t + b.b) / 2
        return any(lb.l - 1 <= cx <= lb.r + 1 and lb.t - 1 <= cy <= lb.b + 1 for lb in links)

    from docling_core.types.doc.document import ContentLayer

    items, furniture = [], []
    for item, _ in result.document.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
        if not hasattr(item, "text") or not item.prov:
            continue
        prov = item.prov[0]
        page = pages.get(prov.page_no)
        if page is None:
            continue
        box = prov.bbox.to_top_left_origin(page_height=page.size.height)
        entry = (prov.page_no, box.t, box.l, item, box)
        if str(getattr(item, "content_layer", "")).lower().endswith("furniture"):
            furniture.append(entry)
        else:
            items.append(entry)

    # Docling files a document title set in the page-header region as furniture
    # (sample 10: 'DATA MANAGEMENT' -> page_header) alongside real running
    # headers and page numbers. Keep a page-1 header that sits above the first
    # body block and whose text recurs on no other page; drop the rest.
    body_top_p1 = min((t for pno, t, _, _, _ in items if pno == 1), default=None)
    seen_elsewhere = {re.sub(r"\s+", " ", it.text).strip().lower()
                      for pno, _, _, it, _ in furniture if pno != 1}
    for entry in furniture:
        pno, top, _, it, _ = entry
        if (pno == 1 and str(it.label) == "page_header" and body_top_p1 is not None
                and top < body_top_p1
                and re.sub(r"\s+", " ", it.text).strip().lower() not in seen_elsewhere):
            items.append(entry)
    items.sort(key=lambda t: (t[0], round(t[1]), t[2]))

    blocks: list[str] = []
    for page_no, _, _, item, box in items:
        prov = item.prov[0]
        inside = [(w, b) for w, b in page_words[prov.page_no]
                  if b.l >= box.l - 1 and b.r <= box.r + 1 and b.t >= box.t - 1 and b.b <= box.b + 1]
        if not inside:
            blocks.append(re.sub(r"[ \t]{2,}", " ", item.text).strip())
            continue
        is_heading = str(item.label) == "section_header"
        # group into lines by vertical position, left to right within a line
        inside.sort(key=lambda wb: (round(wb[1].t), wb[1].l))
        lines: list[list] = []
        for w, b in inside:
            if lines and abs(b.t - lines[-1][0][1].t) <= 2.0:
                lines[-1].append((w, b))
            else:
                lines.append([(w, b)])
        flagged = []
        for line in lines:
            line.sort(key=lambda wb: wb[1].l)
            flagged.append([(w.text, *_word_flags(w.text, getattr(w, "font_name", ""),
                                                   w.rect.height, body_font, body_size,
                                                   in_link(b, page_links[prov.page_no])))
                            for w, b in line])
        # Docling's heading label is a layout signal the fonts may not carry
        # (samples 3, 9 mark headings by size, which the size rule catches; a
        # heading in the body font gets nothing). Add it only in that case.
        font_says_nothing = not any(e or i for ln in flagged for _, e, i, _ in ln)
        if is_heading and font_says_nothing:
            flagged = [[(t, True, i, u) for t, _, i, u in ln] for ln in flagged]
        blocks.append("\n".join(_render_line(ln) for ln in flagged))

    text = "\n".join(b for b in blocks if b)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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
    force_ocr:
        OCR every page (``OcrMode.FULL_PAGE``) instead of trusting the PDF's
        text layer. Needed for documents whose embedded fonts carry no
        character-to-text mapping — sample 11's text layer extracts as
        ``(cid:…)`` garbage while its pages read fine — because Docling's
        auto mode only OCRs pages with *no* text layer, and a garbage layer
        counts as one. Slower (~10 s/page) and OCR loses the font names the
        native text builder uses, so bold/italic markers largely disappear;
        use it when the text layer is broken, not by default.
    """

    name = "docling"          # stage-1 directory the side files go to
    source = "native"         # "native": build the text from the page cells;
                              # "markdown": translate export_to_markdown() (the
                              # earlier, weaker version — kept for comparison)

    def __init__(self, save_native: bool = False, native_images: bool = True,
                 force_ocr: bool = False) -> None:
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
        if force_ocr:
            from docling.datamodel.pipeline_options import OcrMode
            pipeline_options.ocr_options.mode = OcrMode.FULL_PAGE
        if save_native or self.source == "native":
            # Docling discards the parsed pages (cells, fonts, links) once the
            # document is assembled unless told to keep them.
            pipeline_options.generate_parsed_pages = True
        if save_native:
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
        if self.source == "native":
            return [{"text": native_marked_text(result)}]
        return [{"text": markdown_to_marked_text(markdown)}]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_side_file(self, pdf_path: Path, suffix: str, content: str) -> None:
        """Write ``<stem>.<suffix>`` next to the cached stage-1 JSON. Best-effort:
        extraction must not fail because a side artifact could not be written."""
        try:
            from ..core.paths import EXTRACTED_DIR
            out = EXTRACTED_DIR / self.name / f"{pdf_path.stem}.{suffix}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
        except OSError:
            pass

