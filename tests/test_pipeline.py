"""Tests for dmpbridge.core.pipeline.run_and_save().

This is the helper both batch runners (dmpbridge-wholedoc and
dmpbridge-experiment) share, replacing what used to be two independently
maintained copies of the same save-and-convert logic.
"""
import json

from dmpbridge.core.pipeline import run_and_save


class _FakeStrategy:
    """Minimal stand-in for WholeDocStrategy — no PDF or Ollama needed."""

    def run(self, pdf_path):
        return [
            {"text": "My DMP", "label": "title", "page": 1},
            {"text": "Sec 1", "label": "section.title", "page": 1},
            {"text": "The answer.", "label": "answer.text", "page": 1},
        ]


def test_returns_none_when_pdf_missing(tmp_path):
    result = run_and_save(
        _FakeStrategy(),
        pdf_path=tmp_path / "does_not_exist.pdf",
        out_path=tmp_path / "out.json",
        struct_path=tmp_path / "out_structured.json",
    )
    assert result is None
    assert not (tmp_path / "out.json").exists()


def test_saves_flat_and_structured_json(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")  # only existence is checked
    out_path    = tmp_path / "sub" / "out.json"
    struct_path = tmp_path / "sub" / "out_structured.json"

    blocks = run_and_save(_FakeStrategy(), pdf_path, out_path, struct_path)

    assert blocks is not None
    assert out_path.exists()
    assert struct_path.exists()

    saved_blocks = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved_blocks == blocks

    structured = json.loads(struct_path.read_text(encoding="utf-8"))
    assert structured["narrative"]["template"]["title"] == "My DMP"
    assert structured["narrative"]["template"]["section"][0]["title"] == "Sec 1"


def test_struct_path_is_optional(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    out_path = tmp_path / "out.json"

    blocks = run_and_save(_FakeStrategy(), pdf_path, out_path, struct_path=None)

    assert blocks is not None
    assert out_path.exists()
    assert not (tmp_path / "out_structured.json").exists()
