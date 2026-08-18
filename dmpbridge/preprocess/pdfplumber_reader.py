"""Extract a whole-document text blob from a PDF, with visual-signal markers.

Ported from ``notebooks/with-pdfplumber-visual-signals.ipynb``. Unlike the
line-by-line, per-block extraction this module used to do, this reads the
entire PDF and returns one text string per document: each word visually
emphasized relative to the document's own body-text baseline (a different,
non-italic font, or a meaningfully larger size) is wrapped in ``**...**``,
and each italic word in ``_..._``. Headings and questions typically show up
this way even without an explicit "is this a heading" rule, so the marker
is passed to the LLM as supporting evidence rather than encoded as a
separate ``is_bold``/``is_italic`` field the way the old per-line blocks did.

This is a known trade-off, not an oversight: a PDF that renders text twice
in offset layers (some PDFs' fake-bold trick) will still double characters
here, since no char-level deduplication runs on this path.
"""
import re
from collections import Counter
from pathlib import Path
from typing import Union

import pdfplumber


def get_body_font_profile(pdf):
    """Scan all characters in the PDF and determine the dominant font name and
    size, i.e. what "normal paragraph text" looks like in this document.
    Headers/labels/emphasis are then detected as deviations from this
    baseline, rather than by matching a specific font-naming pattern.
    """
    font_sizes = Counter()
    font_names = Counter()

    for page in pdf.pages:
        for char in page.chars:
            font_sizes[round(char["size"], 1)] += 1
            font_names[char["fontname"]] += 1

    if not font_sizes or not font_names:
        # No character-level data available (e.g. scanned/image PDF) -
        # caller should fall back to plain extract_text() in this case.
        return None, None

    body_size = font_sizes.most_common(1)[0][0]
    body_font = font_names.most_common(1)[0][0]
    return body_size, body_font


def is_emphasized(char, body_size, body_font):
    """True if a character looks visually emphasized relative to the document's
    own body text - either a different font family (bold/weight variants
    usually show up here) or a meaningfully larger size.
    """
    different_font = char["fontname"] != body_font
    is_italic_variant = any(
        tag in char["fontname"] for tag in ("Italic", "Oblique", "italic", "oblique")
    )
    looks_bold = different_font and not is_italic_variant
    looks_bigger = char["size"] > body_size + 0.5
    return looks_bold or looks_bigger


def is_italic(char, body_font):
    return any(
        tag in char["fontname"] for tag in ("Italic", "Oblique", "italic", "oblique")
    )


def _word_is_emphasized(word_chars, body_size, body_font):
    """A word counts as emphasized if a majority of its characters are."""
    if not word_chars:
        return False
    flags = [is_emphasized(c, body_size, body_font) for c in word_chars]
    return sum(flags) > len(flags) / 2


def _word_is_italic(word_chars, body_font):
    if not word_chars:
        return False
    flags = [is_italic(c, body_font) for c in word_chars]
    return sum(flags) > len(flags) / 2


def _dedup_consecutive_words(words: list[dict]) -> list[dict]:
    """Drop a word that repeats the immediately preceding word on the same
    line, ignoring case and punctuation.

    Some PDFs render each word twice in slightly offset layers, so pdfplumber
    reports both copies (e.g. "Roles Roles and and responsibilities"). Ported
    from the old text_cleaner.clean_repeated_words(), but run here on the raw
    word list *before* **bold**/_italic_ markers are inserted around them —
    doing it after would let a bare "**"/"_" marker token collide with an
    adjacent punctuation-only word (both normalize to "") and get dropped as
    a false duplicate, corrupting the marker pairing.
    """
    deduped = []
    previous_norm = None
    for w in words:
        norm = re.sub(r"[^\w]", "", w["text"]).lower()
        if norm != previous_norm:
            deduped.append(w)
        previous_norm = norm
    return deduped


def _extract_page_lines(page, body_size, body_font):
    """Rebuild each line of the page from character-level data, wrapping
    emphasized runs in **...** and italic runs in _..._ so the LLM can use
    them as classification signals downstream.
    """
    chars = page.chars
    if not chars:
        return []

    # Group chars into words using pdfplumber's own word extraction for
    # correct spacing/boundaries, then map each word back to its chars
    # (by matching x0/top) to evaluate emphasis per word.
    words = page.extract_words(extra_attrs=["fontname", "size"])

    # Build a lookup of chars by rounded (top) position for fast grouping
    # into lines; pdfplumber words already carry "top" for this purpose.
    lines_map = {}
    for w in words:
        top_key = round(w["top"], 1)
        lines_map.setdefault(top_key, []).append(w)

    # For emphasis, re-derive per-word char list by spatial overlap.
    def chars_for_word(w):
        return [
            c
            for c in chars
            if c["top"] >= w["top"] - 1
            and c["bottom"] <= w["bottom"] + 1
            and c["x0"] >= w["x0"] - 0.5
            and c["x1"] <= w["x1"] + 0.5
        ]

    output_lines = []
    for top_key in sorted(lines_map.keys()):
        line_words = sorted(lines_map[top_key], key=lambda w: w["x0"])
        line_words = _dedup_consecutive_words(line_words)
        rendered = []
        open_bold, open_italic = False, False

        for w in line_words:
            wchars = chars_for_word(w)
            emph = _word_is_emphasized(wchars, body_size, body_font)
            ital = _word_is_italic(wchars, body_font)

            # close markers that no longer apply
            if open_italic and not ital:
                rendered.append("_")
                open_italic = False
            if open_bold and not emph:
                rendered.append("**")
                open_bold = False

            # open markers that newly apply
            if emph and not open_bold:
                rendered.append("**")
                open_bold = True
            if ital and not open_italic:
                rendered.append("_")
                open_italic = True

            rendered.append(w["text"])

        if open_italic:
            rendered.append("_")
        if open_bold:
            rendered.append("**")

        line_text = " ".join(rendered)
        # clean up marker artifacts from consecutive open/close with no content between
        line_text = line_text.replace("** **", " ").replace("_ _", " ")
        output_lines.append(line_text)

    return output_lines


def extract_text_for_llm(pdf_path: Union[str, Path], join_pages_with: str = "\n\n") -> str:
    """Full pipeline: given a PDF path, return a single text string with
    **bold** and _italic_ markers preserved, suitable for passing to a
    whole-document classification call.

    Falls back to plain text extraction (no formatting markers) for any
    page where character-level font data isn't available (e.g. a scanned
    or image-only page), so the pipeline never silently drops content.

    Also collapses runs of excess whitespace and blank lines — the other
    half of what the old text_cleaner.clean_blocks() did. Safe to apply to
    the marker-laden text since it only touches spaces/tabs/newlines, never
    word tokens.
    """
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        body_size, body_font = get_body_font_profile(pdf)

        for page in pdf.pages:
            if body_size is None:
                # No usable font metadata anywhere in the doc - plain fallback
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                continue

            lines = _extract_page_lines(page, body_size, body_font)
            if lines:
                pages_text.append("\n".join(lines))
            else:
                # This page had no extractable words (e.g. scanned image) -
                # fall back to plain extraction so we don't lose the page.
                text = page.extract_text()
                if text:
                    pages_text.append(text)

    result = join_pages_with.join(pages_text)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
