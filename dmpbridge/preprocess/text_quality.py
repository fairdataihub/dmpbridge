"""Detect stage-1 text that is not readable language.

A PDF can have a text layer whose embedded fonts carry no character-to-text
mapping. Extraction then *succeeds* — pdfplumber returns ``(cid:15)(cid:5)…``
tokens, Docling's parser returns shifted mojibake — and the model, given
garbage, invents a fluent document that is in no way the input (observed
directly on sample 11: gemma produced "The quick brown fox…" and lorem-ipsum
paragraphs as the "content" of an NSF plan). Nothing errors, so the only
defence is to look at the text before spending a model call on it.

``looks_garbled`` is deliberately crude: two cheap checks that both sit far
away from every clean document in the corpus (letter ratios 0.78–0.84 across
all extractors' stage-1 texts) and far away on the two failure modes seen.
It returns a human-readable reason, or ``None`` when the text looks fine.
"""
from __future__ import annotations

import re

# Raw character-id tokens: pdfplumber's output when a font has no Unicode map.
_CID = re.compile(r"\(cid:\d+\)")

# Fraction of non-whitespace characters that are ASCII letters, below which
# text does not look like English prose. Clean corpus texts measure 0.78-0.84;
# sample 11's mojibake via Docling measures ~0.35; the threshold splits them
# with a wide margin on both sides.
LETTER_RATIO_FLOOR = 0.60

# U+FFFD is what decoders substitute for bytes they cannot map; more than a
# handful means the text layer's encoding is broken, not the odd stray glyph.
REPLACEMENT_CHAR_CEILING = 10

# Non-whitespace characters per page below which a "digital" PDF most likely
# has no real text layer (a scanned page extracts as nothing, or a few stray
# marks). The shortest clean corpus document measures well over 1,000/page.
MIN_CHARS_PER_PAGE = 150


def looks_garbled(text: str, n_pages: int | None = None) -> str | None:
    """Return a reason string if *text* does not look like readable language.

    ``n_pages``, when known, enables the characters-per-page floor — the
    signature of a scanned document whose "extraction" is a near-empty string.
    """
    if not text or not text.strip():
        return "empty extraction"

    cid_hits = len(_CID.findall(text))
    if cid_hits > 10:
        return (f"{cid_hits} raw (cid:NN) tokens — the PDF's fonts have no "
                f"character-to-text mapping")

    replacements = text.count("�")
    if replacements > REPLACEMENT_CHAR_CEILING:
        return (f"{replacements} replacement characters (U+FFFD) — the text "
                f"layer's encoding could not be decoded")

    stripped = re.sub(r"\s", "", text)
    if stripped:
        letters = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
        ratio = letters / len(stripped)
        if ratio < LETTER_RATIO_FLOOR:
            return (f"only {ratio:.0%} of characters are letters "
                    f"(clean documents measure ~80%) — likely a broken text layer")

    if n_pages and len(stripped) / n_pages < MIN_CHARS_PER_PAGE:
        return (f"{len(stripped)} characters across {n_pages} page(s) — "
                f"too little text for a digital PDF; likely scanned")

    return None
