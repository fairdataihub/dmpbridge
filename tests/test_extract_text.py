"""WholeDocStrategy.extract_text(): the shared text-extraction step.

The narrative pipeline (run) and the RDA extraction both read a plan's text
through extract_text(), so they see the same text: stage-1 cache, garbled-text
check, fallback. These tests pin that split - run() is extract_text() followed
by the labeling call - without a PDF or an Ollama server.
"""
import json

from pathlib import Path

import pytest

from dmpbridge.strategies.wholedoc import WholeDocStrategy

PDF = Path("sample1.pdf")


class FakeExtractor:
    name = "fake"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def extract(self, pdf_path):
        self.calls += 1
        return [{"text": self.text}] if self.text else []


def make(text, cache_dir=None):
    s = WholeDocStrategy.__new__(WholeDocStrategy)        # no model backend needed
    s.model, s.extractor_name, s._fallbacks = "fake-model", "fake", []
    s.cache_dir = cache_dir
    s._extractor = FakeExtractor(text)
    return s


def test_extract_text_returns_the_full_text():
    assert make("A clean plan about data.").extract_text(PDF) == "A clean plan about data."


def test_extract_text_uses_the_stage1_cache(tmp_path):
    (tmp_path / "sample1.json").write_text(json.dumps([{"text": "cached text"}]), encoding="utf-8")
    s = make("fresh text", cache_dir=tmp_path)
    assert s.extract_text(PDF) == "cached text"
    assert s._extractor.calls == 0


def test_extract_text_is_empty_for_a_document_without_text():
    assert make("").extract_text(PDF) == ""


def test_run_is_extract_text_then_labeling(monkeypatch):
    s = make("Plan text.")
    seen = []
    monkeypatch.setattr(s, "classify_entire_document", lambda text: seen.append(text) or [{"text": text, "label": "x"}])
    assert s.run(PDF) == [{"text": "Plan text.", "label": "x"}]
    assert seen == ["Plan text."]


def test_run_skips_labeling_when_there_is_no_text(monkeypatch):
    s = make("")
    monkeypatch.setattr(s, "classify_entire_document", lambda text: pytest.fail("labeling called"))
    assert s.run(PDF) == []
