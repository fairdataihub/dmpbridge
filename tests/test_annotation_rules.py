"""Tests for dmpbridge.evaluation.annotation_rules (Path B: new-annotation eval).

apply_new_annotation_rules() itself is exercised against real ground truth
here (it was derived from and validated against that data in
notebooks/annotation_conversion_test.ipynb); convert_tag_to_final() and
load_method_new() are tested against temporary directories via monkeypatch so
they don't touch real project data.
"""
import json

import pytest

import dmpbridge.evaluation.annotation_rules as ar


# ── apply_new_annotation_rules ───────────────────────────────────────────────

def test_backfills_empty_question_from_section_title():
    data = {"narrative": {"template": {"title": "Doc", "section": [{
        "title": "Sec 1",
        "question": [{"text": "", "answer": {"json": {"answer": "content"}}}],
    }]}}}
    out = ar.apply_new_annotation_rules(data)
    q = out["narrative"]["template"]["section"][0]["question"][0]
    assert q["text"] == "Sec 1"


def _doc(title="", sec_title="", sec_desc="", q_text="", answer="content"):
    return {"narrative": {"template": {"title": title, "section": [{
        "title": sec_title,
        "description": sec_desc,
        "question": [{"text": q_text, "answer": {"json": {"answer": answer}}}],
    }]}}}


def _fields(out):
    t = out["narrative"]["template"]
    s = t["section"][0]
    return t["title"], s["title"], s["question"][0]["text"]


def test_rows_9_10_copy_title_into_question_only():
    """Rows 9/10: section.title stays empty and the document title is kept.

    Rules.xlsx originally said to fill section.title as well; it was corrected
    on 5 Aug 2026 to match the reference files for samples 4 and 7, the only
    two documents that reach this branch.
    """
    title, sec_title, q_text = _fields(ar.apply_new_annotation_rules(_doc(title="Doc Title")))
    assert q_text == "Doc Title"
    assert sec_title == ""          # not filled
    assert title == "Doc Title"     # not cleared


def test_row_2_copies_section_description_into_both():
    """Row 2: description is the only source, so it fills question and section title."""
    _, sec_title, q_text = _fields(
        ar.apply_new_annotation_rules(_doc(sec_desc="Funder guidance text")))
    assert q_text == "Funder guidance text"
    assert sec_title == "Funder guidance text"


def test_rows_5_6_13_14_copy_question_into_empty_section_title():
    """Rows 5/6/13/14: the reverse direction — question.text fills section.title."""
    for title in ("", "Doc Title"):
        for desc in ("", "Some guidance"):
            _, sec_title, q_text = _fields(ar.apply_new_annotation_rules(
                _doc(title=title, sec_desc=desc, q_text="A. Types of data")))
            assert sec_title == "A. Types of data", f"title={title!r} desc={desc!r}"
            assert q_text == "A. Types of data"


def test_document_title_outranks_section_description():
    """Rows 9/10 beat row 2 when both could fill an empty pair."""
    _, sec_title, q_text = _fields(ar.apply_new_annotation_rules(
        _doc(title="Doc Title", sec_desc="Guidance")))
    assert q_text == "Doc Title"
    assert sec_title == ""


def test_document_title_used_once_only():
    """A second empty section must not be filled from the same document title."""
    data = {"narrative": {"template": {"title": "Doc Title", "section": [
        {"title": "", "description": "",
         "question": [{"text": "", "answer": {"json": {"answer": "a"}}}]},
        {"title": "", "description": "",
         "question": [{"text": "", "answer": {"json": {"answer": "b"}}}]},
    ]}}}
    sections = ar.apply_new_annotation_rules(data)["narrative"]["template"]["section"]
    assert sections[0]["question"][0]["text"] == "Doc Title"
    assert sections[1]["question"][0]["text"] == ""


def test_rows_7_8_15_16_leave_both_populated_fields_alone():
    _, sec_title, q_text = _fields(ar.apply_new_annotation_rules(
        _doc(title="Doc", sec_title="Sec 1", q_text="Q 1")))
    assert (sec_title, q_text) == ("Sec 1", "Q 1")


def test_row_1_leaves_everything_empty_when_nothing_to_copy():
    _, sec_title, q_text = _fields(ar.apply_new_annotation_rules(_doc()))
    assert (sec_title, q_text) == ("", "")


def test_leaves_non_empty_question_text_untouched():
    data = {"narrative": {"template": {"title": "Doc", "section": [{
        "title": "Sec 1",
        "question": [{"text": "Already filled", "answer": {"json": {"answer": "x"}}}],
    }]}}}
    out = ar.apply_new_annotation_rules(data)
    assert out["narrative"]["template"]["section"][0]["question"][0]["text"] == "Already filled"


def test_reproduces_real_new_annotation_exactly_for_non_collapse_samples():
    """Regression test: this exact match was how the rule was validated originally."""
    from dmpbridge.evaluation.evaluate import MANUAL_DIR

    for n in [1, 3, 4, 7]:
        old = json.loads((MANUAL_DIR / f"sample{n}_old_dmp.json").read_text(encoding="utf-8"))
        new = json.loads(ar.resolve_new_gt_path(n).read_text(encoding="utf-8"))
        got = ar.apply_new_annotation_rules(old)
        assert got == new, f"sample{n} no longer matches the real new-version ground truth"


# ── resolve_new_gt_path ──────────────────────────────────────────────────────

def test_resolve_new_gt_path_finds_both_naming_patterns():
    # samples 1,3,4,7 use sampleN_dmp_new.json; samples 5,6,8,9,10 use dmp_sampleN_new.json
    for n in [1, 5]:
        path = ar.resolve_new_gt_path(n)
        assert path.exists()
        assert str(n) in path.stem


def test_resolve_new_gt_path_raises_for_unknown_sample():
    with pytest.raises(FileNotFoundError):
        ar.resolve_new_gt_path(9999)


# ── convert_tag_to_final / load_method_new (isolated via monkeypatch) ───────

def _write_structured(path, title, section_title, question_text, answer):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"narrative": {"template": {
        "title": title,
        "section": [{
            "title": section_title,
            "question": [{"text": question_text, "answer": {"json": {"answer": answer}}}],
        }],
    }}}), encoding="utf-8")


def test_convert_tag_to_final_writes_rule_converted_output(tmp_path, monkeypatch):
    """Stage 3 -> stage 4, with identical filenames in each stage directory."""
    structured_dir = tmp_path / "3_structured"
    final_dir      = tmp_path / "4_final"
    monkeypatch.setattr(ar._paths, "STRUCTURED_DIR", structured_dir)
    monkeypatch.setattr(ar, "FINAL_DIR", final_dir)

    _write_structured(structured_dir / "my-tag" / "sample1.json",
                       title="Doc", section_title="Sec 1", question_text="", answer="content")

    n = ar.convert_tag_to_final("my-tag")

    assert n == 1
    out = json.loads((final_dir / "my-tag" / "sample1.json").read_text(encoding="utf-8"))
    assert out["narrative"]["template"]["section"][0]["question"][0]["text"] == "Sec 1"


def test_load_method_scores_final_json_against_resolved_gold(tmp_path, monkeypatch):
    final_dir = tmp_path / "4_final"
    monkeypatch.setattr(ar, "FINAL_DIR", final_dir)

    _write_structured(final_dir / "my-tag" / "sample1.json",
                       title="Doc", section_title="Sec 1",
                       question_text="Sec 1", answer="The answer text.")

    def fake_resolve(n):
        gold_path = tmp_path / f"gold{n}.json"
        gold_path.write_text(json.dumps({"narrative": {"template": {
            "title": "Doc",
            "section": [{
                "title": "Sec 1",
                "question": [{"text": "Sec 1", "answer": {"json": {"answer": "The answer text."}}}],
            }],
        }}}), encoding="utf-8")
        return gold_path

    monkeypatch.setattr(ar, "resolve_new_gt_path", fake_resolve)

    df, conf, errors = ar.load_method_new("my-tag")

    assert df is not None
    assert int(df["correct"].sum()) == int(df["total"].sum())  # perfect match
    assert conf["answer.text"]["answer.text"] == 1


def test_load_method_returns_none_for_missing_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "FINAL_DIR", tmp_path / "labeled_final")
    df, conf, errors = ar.load_method_new("does-not-exist")
    assert df is None
    assert conf is None
    assert errors is None
