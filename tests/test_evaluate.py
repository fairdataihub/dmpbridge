"""Tests for the matching/scoring core of dmpbridge.evaluation.evaluate.

These lock down the behaviors this codebase has already been debugged
against once (the run_single crash on non-structured input, the
confusion-matrix / missed-items / error-table consistency fix, and the
precision-aware micro_prf1 metric) so they can't silently regress.
"""
import json

import pytest

from dmpbridge.evaluation.evaluate import (
    _confusion_from_match,
    _match_structured,
    containment,
    extract_gold,
    gold_metrics,
    match,
    micro_prf1,
    tokenize,
)


# ── tokenize / containment ──────────────────────────────────────────────────

def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Data, Management!  Plan.") == {"data", "management", "plan"}


def test_containment_empty_block_is_zero():
    assert containment(set(), {"a", "b"}) == 0.0


def test_containment_fraction_of_block_tokens_in_gold():
    block = {"a", "b", "c", "d"}
    gold  = {"a", "b", "x", "y"}
    assert containment(block, gold) == 0.5


# ── extract_gold ─────────────────────────────────────────────────────────────

def _structured(title="Doc", sections=None):
    return {
        "narrative": {
            "template": {
                "title": title,
                "section": sections or [],
            }
        }
    }


def test_extract_gold_basic_pairs():
    data = _structured(sections=[{
        "title": "Sec 1",
        "description": "A description.",
        "question": [{"text": "A question?", "answer": {"json": {"answer": "An answer."}}}],
    }])
    pairs = extract_gold_from_dict(data)
    assert ("Doc", "title") in pairs
    assert ("Sec 1", "section.title") in pairs
    assert ("A description.", "section.description") in pairs
    assert ("A question?", "question.text") in pairs
    assert ("An answer.", "answer.text") in pairs


def test_extract_gold_skips_question_text_matching_section_title():
    """PM rule: question.text duplicating its own section.title is not double-counted."""
    data = _structured(sections=[{
        "title": "Sec 1",
        "question": [{"text": "Sec 1", "answer": {"json": {"answer": "ans"}}}],
    }])
    pairs = extract_gold_from_dict(data)
    labels = [lbl for _, lbl in pairs]
    assert labels.count("question.text") == 0


def test_extract_gold_raises_clear_error_on_non_structured_input(tmp_path):
    """Regression test for the run_single crash on raw (non-structured) block JSON."""
    bad_path = tmp_path / "sample1.json"
    bad_path.write_text(json.dumps([{"text": "a", "label": "answer.text"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="structured DMP JSON"):
        extract_gold(bad_path)


def extract_gold_from_dict(data: dict, tmp_path=None):
    """Helper: run extract_gold's logic on an in-memory dict via a temp file."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sample.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return extract_gold(p)


# ── match() ──────────────────────────────────────────────────────────────────

def test_match_returns_label_when_fully_contained():
    """The default is exact containment: every predicted word must be in the gold."""
    gold_pairs = [("the quick brown fox jumps", "answer.text")]
    assert match("the quick brown fox", gold_pairs) == "answer.text"


def test_partial_overlap_needs_an_explicit_looser_threshold():
    """4 of 5 words in common is 0.8 containment.

    That matched under the project's old 0.75 default and does not under the
    current 1.0 one — so the looser rule must now be asked for by name. Pinning
    both here means neither can change silently again.
    """
    gold_pairs = [("the quick brown fox", "answer.text")]
    assert match("the quick brown fox jumps", gold_pairs) is None
    assert match("the quick brown fox jumps", gold_pairs, threshold=0.75) == "answer.text"


def test_match_returns_none_below_threshold():
    gold_pairs = [("the quick brown fox", "answer.text")]
    assert match("completely unrelated text here", gold_pairs) is None


def test_match_returns_none_for_empty_block():
    assert match("", [("something", "answer.text")]) is None


# ── _match_structured / _confusion_from_match ────────────────────────────────

def test_match_structured_exact_match_is_true_positive(tmp_path):
    pred_path = tmp_path / "pred_structured.json"
    pred_path.write_text(json.dumps(_structured(title="", sections=[{
        "title": "",
        "question": [{"text": "Q", "answer": {"json": {"answer": "The answer text."}}}],
    }])), encoding="utf-8")

    gold_pairs = [("The answer text.", "answer.text"), ("Q", "question.text")]
    records, no_gold = _match_structured(pred_path, gold_pairs)

    assert len(records) == 2
    assert no_gold == []  # every predicted item was claimed

    confusion = _confusion_from_match(records, no_gold)
    assert confusion["answer.text"]["answer.text"] == 1
    assert confusion["question.text"]["question.text"] == 1


def test_match_structured_mislabeled_item(tmp_path):
    pred_path = tmp_path / "pred_structured.json"
    pred_path.write_text(json.dumps(_structured(sections=[{
        "title": "Sec 1",
        "question": [{"text": "The real answer text.", "answer": {"json": {"answer": ""}}}],
    }])), encoding="utf-8")

    gold_pairs = [("The real answer text.", "answer.text")]
    records, no_gold = _match_structured(pred_path, gold_pairs)
    confusion = _confusion_from_match(records, no_gold)

    # Gold wanted answer.text but the model put the same text under question.text.
    assert confusion["answer.text"]["question.text"] == 1
    assert confusion["answer.text"].get("answer.text", 0) == 0


def test_match_structured_missed_gold_item(tmp_path):
    pred_path = tmp_path / "pred_structured.json"
    pred_path.write_text(json.dumps(_structured(sections=[])), encoding="utf-8")

    gold_pairs = [("Never predicted.", "answer.text")]
    records, no_gold = _match_structured(pred_path, gold_pairs)
    confusion = _confusion_from_match(records, no_gold)

    assert confusion["answer.text"]["__missed__"] == 1


def test_match_structured_spurious_prediction_goes_to_no_gold(tmp_path):
    pred_path = tmp_path / "pred_structured.json"
    pred_path.write_text(json.dumps(_structured(sections=[{
        "title": "Sec 1",
        "question": [{"text": "Hallucinated content nobody asked for.",
                       "answer": {"json": {"answer": ""}}}],
    }])), encoding="utf-8")

    gold_pairs: list[tuple[str, str]] = []  # nothing in gold to match against
    records, no_gold = _match_structured(pred_path, gold_pairs)
    confusion = _confusion_from_match(records, no_gold)

    assert confusion["__no_gold__"]["question.text"] == 1


def test_match_structured_does_not_reuse_a_claimed_prediction(tmp_path):
    """Greedy matching: once a predicted item is claimed by one gold item, a
    second gold item can't also claim it, even with a perfect containment score.
    """
    pred_path = tmp_path / "pred_structured.json"
    pred_path.write_text(json.dumps(_structured(title="", sections=[{
        "title": "",
        "question": [{"text": "Shared text", "answer": {"json": {"answer": ""}}}],
    }])), encoding="utf-8")

    gold_pairs = [("Shared text", "question.text"), ("Shared text", "section.title")]
    records, no_gold = _match_structured(pred_path, gold_pairs)

    matched_count = sum(1 for r in records if r["pred_label"] is not None)
    assert matched_count == 1
    assert no_gold == []


# ── micro_prf1 / gold_metrics ────────────────────────────────────────────────

def test_micro_prf1_perfect_score():
    confusion = {"answer.text": {"answer.text": 4}}
    m = micro_prf1(confusion)
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 4, "fp": 0, "fn": 0}


def test_micro_prf1_penalizes_spurious_predictions():
    """This is the exact bug this metric was added to fix: gold_metrics()
    ignores __no_gold__ entirely, so precision loss from hallucinated output
    was invisible before micro_prf1 existed.
    """
    confusion = {
        "answer.text": {"answer.text": 2},
        "__no_gold__": {"answer.text": 8},  # 8 hallucinated, unmatched predictions
    }
    m = micro_prf1(confusion)
    assert m["tp"] == 2
    assert m["fp"] == 8
    assert m["precision"] == pytest.approx(0.2)
    assert m["recall"] == 1.0


def test_micro_prf1_penalizes_missed_gold_items():
    confusion = {"answer.text": {"answer.text": 1, "__missed__": 3}}
    m = micro_prf1(confusion)
    assert m["tp"] == 1
    assert m["fn"] == 3
    assert m["recall"] == 0.25
    assert m["precision"] == 1.0


def test_micro_prf1_empty_confusion_is_zero_not_a_crash():
    m = micro_prf1({})
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


def test_gold_metrics_ignores_no_gold_spurious_predictions():
    """Documents the known difference from micro_prf1: gold_metrics() is
    recall-only and does not penalize hallucinated/spurious predictions.
    """
    confusion = {
        "answer.text": {"answer.text": 2},
        "__no_gold__": {"answer.text": 100},
    }
    correct, covered, total = gold_metrics(confusion)
    assert correct == 2
    assert total == 2  # unaffected by the 100 spurious predictions
