"""Tests for the wrapped-line merger."""
from dmpbridge.preprocess.line_merger import merge_wrapped_lines


RIGHT_MARGIN = 535.0


def block(text, *, page=1, x0=72.0, top=100.0, size=11.0,
          bold=False, italic=False, height=12.0, x1=None):
    """Build a block dict with sane geometry defaults.

    Defaults place the block's right edge at the page margin, i.e. it looks
    like a wrapped body line.  Pass a small ``x1`` to make it look like a
    heading that stops short of the margin.
    """
    return {
        "page": page, "line_order": 1, "text": text,
        "x0": x0, "top": top, "x1": RIGHT_MARGIN if x1 is None else x1,
        "bottom": top + height,
        "avg_font_size": size, "font_names": ["Times"],
        "is_bold": bold, "is_italic": italic, "label": None,
    }


def stack(texts, **kw):
    """Build vertically stacked blocks, one line beneath the next."""
    return [block(t, top=100.0 + i * 14.0, **kw) for i, t in enumerate(texts)]


# ── Core behaviour ────────────────────────────────────────────────────────────

def test_wrapped_paragraph_is_joined():
    blocks = stack([
        "We will store all data in a secure repository",
        "maintained by the university and backed up",
        "nightly, with access limited to approved",
        "published manuscripts.",
    ])
    merged = merge_wrapped_lines(blocks)
    assert len(merged) == 1
    assert merged[0]["text"].startswith("We will store")
    assert merged[0]["text"].endswith("published manuscripts.")


def test_trailing_fragment_is_absorbed():
    """The orphan-tail case this module exists to fix."""
    blocks = stack(["...with access limited to approved", "published manuscripts."])
    assert len(merge_wrapped_lines(blocks)) == 1


def test_completed_sentences_stay_separate():
    blocks = stack(["This is a complete sentence.", "This is another one."])
    assert len(merge_wrapped_lines(blocks)) == 2


def test_lowercase_start_merges_even_after_period():
    """An abbreviation ends in '.' but the next line clearly continues it."""
    blocks = stack(["Data will be shared with the U.S.", "agency within one year."])
    assert len(merge_wrapped_lines(blocks)) == 1


# ── Guards ────────────────────────────────────────────────────────────────────

def test_heading_is_not_merged_into_body():
    blocks = [
        block("Element 1: Data Type", top=100.0, bold=True),
        block("we will generate imaging data", top=114.0, bold=False),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_two_short_headings_are_not_merged():
    """Regression: a document title followed by a section title.

    Neither ends in punctuation, so the 'unfinished' cue alone would join
    them. Both stop far short of the right margin, which is what saves them.
    """
    blocks = [
        block("CPS 2015", top=72.0, x1=128.0),
        block("Roles and responsibilities", top=100.0, x1=211.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_heading_above_body_text_is_not_merged():
    """A short heading followed by a full-width line that starts uppercase."""
    blocks = [
        block("Types of data", top=100.0, x1=180.0),
        block("The Data Management Plan should describe the types", top=114.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


# ── Page boundaries ───────────────────────────────────────────────────────────

def test_paragraph_continuing_across_a_page_break_is_merged():
    """Real case from sample3: a sentence split by a page break."""
    blocks = [
        block("The proposed developments will produce simulation code for controller",
              page=1, top=696.0),
        block("development and testing. These outputs will also be managed.",
              page=2, top=70.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 1


def test_page_break_needs_both_cues():
    """Uppercase start across a page break is a new block, not a continuation."""
    blocks = [
        block("text continuing across", page=1, top=700.0),
        block("The next section begins here", page=2, top=100.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_running_header_is_not_absorbed():
    """A page header starts uppercase and stops short of the margin."""
    blocks = [
        block("...and the results will be published in due", page=1, top=700.0),
        block("Data Management Plan — page 2", page=2, top=40.0, x1=250.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_different_left_edge_is_not_merged():
    blocks = [
        block("first list item that wraps", x0=72.0, top=100.0),
        block("indented sub-item", x0=108.0, top=114.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_large_vertical_gap_is_not_merged():
    blocks = [
        block("end of one paragraph with no period", top=100.0),
        block("start of a distant block", top=400.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_font_size_change_is_not_merged():
    blocks = [
        block("large heading text", top=100.0, size=18.0),
        block("body copy beneath it", top=114.0, size=11.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_bold_label_and_regular_answer_stay_split():
    """_line_to_blocks splits these deliberately; merging must not undo it."""
    blocks = [
        block("Roles:", top=100.0, bold=True),
        block("the PI will oversee data collection", top=100.0, bold=False),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


# ── Metadata ──────────────────────────────────────────────────────────────────

def test_bounding_box_is_widened():
    blocks = [
        block("first line without an ending", x0=72.0, top=100.0),
        block("second line here.", x0=72.0, top=114.0),
    ]
    merged = merge_wrapped_lines(blocks)[0]
    assert merged["top"] == 100.0
    assert merged["bottom"] == 126.0        # 114 + 12 height


def test_font_names_are_unioned():
    a = block("first line without an ending", top=100.0)
    b = block("second line here.", top=114.0)
    b["font_names"] = ["Times", "Times-Italic"]
    merged = merge_wrapped_lines([a, b])[0]
    assert merged["font_names"] == ["Times", "Times-Italic"]


def test_input_blocks_are_not_mutated():
    blocks = stack(["first line without an ending", "second line here."])
    before = blocks[0]["text"]
    merge_wrapped_lines(blocks)
    assert blocks[0]["text"] == before


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_input():
    assert merge_wrapped_lines([]) == []


def test_single_block_passes_through():
    assert len(merge_wrapped_lines([block("only one")])) == 1


def test_already_paragraph_segmented_is_unchanged():
    """Safe to run on Docling/LightOnOCR-style output."""
    blocks = [
        block("A complete paragraph ending properly.", top=100.0),
        block("Another complete paragraph.", top=200.0),
    ]
    assert len(merge_wrapped_lines(blocks)) == 2


def test_missing_geometry_does_not_crash():
    a = block("line without geometry")
    b = block("continuation here.")
    for k in ("x0", "top", "x1", "bottom"):
        a[k] = b[k] = None
    merged = merge_wrapped_lines([a, b])
    assert len(merged) == 1
