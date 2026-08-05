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


def test_copies_document_title_when_section_title_also_empty():
    """The title fills the question and is *kept* on the template.

    It used to be cleared. The new-version ground truth was revised on
    5 Aug 2026 to retain it — see test_reproduces_real_new_annotation_*.
    """
    data = {"narrative": {"template": {"title": "Doc Title", "section": [{
        "title": "",
        "question": [{"text": "", "answer": {"json": {"answer": "content"}}}],
    }]}}}
    out = ar.apply_new_annotation_rules(data)
    template = out["narrative"]["template"]
    assert template["section"][0]["question"][0]["text"] == "Doc Title"
    assert template["title"] == "Doc Title"


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
    llm_dir   = tmp_path / "labeled"
    final_dir = tmp_path / "labeled_final"
    monkeypatch.setattr(ar, "LLM_DIR", llm_dir)
    monkeypatch.setattr(ar, "FINAL_DIR", final_dir)

    _write_structured(llm_dir / "my-tag" / "sample1_structured.json",
                       title="Doc", section_title="Sec 1", question_text="", answer="content")

    n = ar.convert_tag_to_final("my-tag")

    assert n == 1
    out = json.loads((final_dir / "my-tag" / "sample1_structured.json").read_text(encoding="utf-8"))
    assert out["narrative"]["template"]["section"][0]["question"][0]["text"] == "Sec 1"


def test_load_method_scores_final_json_against_resolved_gold(tmp_path, monkeypatch):
    final_dir = tmp_path / "labeled_final"
    monkeypatch.setattr(ar, "FINAL_DIR", final_dir)

    _write_structured(final_dir / "my-tag" / "sample1_structured.json",
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
