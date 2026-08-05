"""Join wrapped lines back into paragraph-level blocks.

pdfplumber segments a PDF by line, not by paragraph, so a single answer
paragraph arrives as several blocks:

    "We will store all data in a secure repository"
    "maintained by the university and backed up"
    "nightly, with access limited to approved"
    "published manuscripts."

The classifier labels each block independently and is usually consistent, but
a short trailing fragment is easy to mislabel on its own.  When that happens
the downstream converter cannot repair it — it only merges neighbours that
already share a label — so the fragment becomes a spurious item *and* the
parent loses its tail, which can drop it below the evaluation's containment
threshold.  One slip costs both a false positive and a false negative.

Merging the lines before classification removes the error class instead of
asking later stages to compensate for it.  Docling and LightOnOCR already
segment by paragraph; this brings pdfplumber into line with them.

Deciding what continues what
----------------------------
The decisive signal is the *right* margin, not the left.  A line that wrapped
runs out to the right edge of the text column; a heading stops well short of
it.  On a typical page here, wrapped body lines end within ~35pt of the
margin while headings fall 300pt+ short — a wide, reliable separation.

A merge therefore needs a continuation cue:

* the next line opens lowercase (strong — it cannot start a new sentence), or
* this line both looks unfinished *and* reaches the right margin

plus agreement on page, font, weight and left edge.  Without the right-margin
test, every short heading looks "unfinished" (headings rarely end in a full
stop) and gets welded onto whatever follows it.

The rule stays conservative: a missed merge is recoverable downstream, but a
wrong merge silently destroys two items at once.
"""
import re

# A line ending in one of these is treated as a completed thought.
_SENTENCE_END = re.compile(r"[.!?:;]['\")\]]?\s*$")

# Tolerances, in PDF points.
_X0_TOLERANCE        = 2.0    # left edges must align this closely
_FONT_SIZE_TOLERANCE = 0.5    # font size must match this closely
_GAP_LINE_HEIGHTS    = 1.5    # vertical gap, as a multiple of line height
_MARGIN_TOLERANCE    = 60.0   # how short of the right margin still counts as "wrapped"


def _looks_unfinished(text: str) -> bool:
    """True when *text* does not end on sentence-final punctuation."""
    return not _SENTENCE_END.search(text.rstrip())


def _starts_lowercase(text: str) -> bool:
    """True when *text* opens with a lowercase letter — a continuation cue."""
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].islower()


def _page_right_margins(blocks: list[dict]) -> dict:
    """Map each page to the rightmost extent of its text."""
    margins: dict = {}
    for b in blocks:
        x1 = b.get("x1")
        if x1 is None:
            continue
        page = b.get("page")
        if x1 > margins.get(page, float("-inf")):
            margins[page] = x1
    return margins


def _reaches_right_margin(block: dict, margins: dict) -> bool:
    """True when *block* runs out to the text column's right edge.

    This is what separates a wrapped body line from a heading.  Blocks with no
    geometry are given the benefit of the doubt, since the textual cues are
    then the only evidence available.
    """
    x1 = block.get("x1")
    margin = margins.get(block.get("page"))
    if x1 is None or margin is None:
        return True
    return (margin - x1) <= _MARGIN_TOLERANCE


def _same_style(a: dict, b: dict) -> bool:
    """True when two blocks share bold/italic state and font size.

    The bold check also keeps the bold-label / regular-answer pair produced by
    ``_line_to_blocks`` apart, which is the intended behaviour — they are meant
    to be classified independently.
    """
    if bool(a.get("is_bold")) != bool(b.get("is_bold")):
        return False
    if bool(a.get("is_italic")) != bool(b.get("is_italic")):
        return False
    sa, sb = a.get("avg_font_size") or 0.0, b.get("avg_font_size") or 0.0
    if sa and sb and abs(sa - sb) > _FONT_SIZE_TOLERANCE:
        return False
    return True


def _same_column(a: dict, b: dict) -> bool:
    """True when two blocks start at the same left edge."""
    xa, xb = a.get("x0"), b.get("x0")
    if xa is None or xb is None:
        return True          # no geometry available — do not block on it
    return abs(xa - xb) <= _X0_TOLERANCE


def _vertically_adjacent(a: dict, b: dict) -> bool:
    """True when *b* sits directly beneath *a* with no section-sized gap."""
    a_bottom, b_top = a.get("bottom"), b.get("top")
    if a_bottom is None or b_top is None:
        return True          # no geometry available — do not block on it
    gap = b_top - a_bottom
    if gap < 0:
        return False         # b is above a — different column or out of order
    line_height = (a.get("bottom") or 0) - (a.get("top") or 0)
    if line_height <= 0:
        line_height = a.get("avg_font_size") or 12.0
    return gap <= line_height * _GAP_LINE_HEIGHTS


def _continues(a: dict, b: dict, margins: dict) -> bool:
    """True when block *b* is a continuation of block *a*."""
    if not _same_style(a, b) or not _same_column(a, b):
        return False

    wrapped   = _looks_unfinished(a.get("text", "")) and _reaches_right_margin(a, margins)
    lowercase = _starts_lowercase(b.get("text", ""))

    if a.get("page") != b.get("page"):
        # Across a page break the vertical test is meaningless and running
        # headers may intervene, so demand both cues before joining.
        return wrapped and lowercase

    if not _vertically_adjacent(a, b):
        return False
    return wrapped or lowercase


def _absorb(target: dict, extra: dict) -> None:
    """Append *extra* onto *target* in place, widening the bounding box."""
    target["text"] = f"{target['text'].rstrip()} {extra['text'].lstrip()}".strip()

    for key, pick in (("x0", min), ("top", min), ("x1", max), ("bottom", max)):
        tv, ev = target.get(key), extra.get(key)
        if tv is not None and ev is not None:
            target[key] = round(pick(tv, ev), 2)
        elif target.get(key) is None:
            target[key] = ev

    for name in extra.get("font_names") or []:
        if name not in target.setdefault("font_names", []):
            target["font_names"].append(name)


def merge_wrapped_lines(blocks: list[dict]) -> list[dict]:
    """Return *blocks* with wrapped lines joined into paragraph-level blocks.

    Input blocks are left untouched; merged copies are returned.  Blocks that
    are not continuations pass through unchanged, so this is safe to run on
    output that is already paragraph-segmented.
    """
    if not blocks:
        return []

    margins = _page_right_margins(blocks)
    merged: list[dict] = [dict(blocks[0])]
    for block in blocks[1:]:
        if _continues(merged[-1], block, margins):
            _absorb(merged[-1], block)
        else:
            merged.append(dict(block))
    return merged


__all__ = ["merge_wrapped_lines"]
