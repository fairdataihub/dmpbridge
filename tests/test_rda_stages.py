"""Stages 5 and 6 (dmpbridge.rda.stages).

No Ollama server and no PDF: the model call and the text extraction are faked, and
every output directory is redirected to a temporary folder.
"""
import json

import pytest

from dmpbridge.core import paths
from dmpbridge.rda import RdaExtractionError
from dmpbridge.rda import stages


class FakeStrategy:
    def __init__(self, text="Plan text."):
        self.text, self.calls = text, 0

    def extract_text(self, pdf_path):
        self.calls += 1
        return self.text


@pytest.fixture
def out(tmp_path, monkeypatch):
    for name in ("FINAL_DIR", "RDA_DIR", "MERGED_DIR"):
        monkeypatch.setattr(paths, name, tmp_path / name)
    pdf = tmp_path / "sample3.pdf"
    pdf.write_bytes(b"%PDF")
    narrative = paths.final_path(paths.make_tag("gemma4:e4b", "pdfplumber"), 3)
    narrative.parent.mkdir(parents=True)
    narrative.write_text(json.dumps({"narrative": {"title": "N"}}), encoding="utf-8")
    return pdf


def run(pdf, strategy, rda_model="llama3.1:8b"):
    return stages.run_rda_stages(strategy, pdf, 3, rda_model, "gemma4:e4b", "pdfplumber")


def save_rda(doc, rda_model):
    """A stage-5 file as an earlier run with *rda_model* would have left it."""
    rda_out = paths.rda_path(3)
    rda_out.parent.mkdir(parents=True, exist_ok=True)
    rda_out.write_text(json.dumps(doc), encoding="utf-8")
    paths.run_info_path(rda_out).write_text(json.dumps(stages.run_record(rda_model, "pdfplumber")),
                                            encoding="utf-8")


def test_writes_rda_and_merged_without_model_folders(out, monkeypatch):
    monkeypatch.setattr(stages, "extract_rda", lambda text, model, host: ({"dmp": {"title": text}}, {}))
    rda_out, merged_out = run(out, FakeStrategy())
    assert rda_out == paths.RDA_DIR / "sample3.json"
    assert merged_out == paths.MERGED_DIR / "sample3.json"
    assert json.loads(rda_out.read_text(encoding="utf-8")) == {"dmp": {"title": "Plan text."}}
    assert list(json.loads(merged_out.read_text(encoding="utf-8"))) == ["dmp", "narrative"]
    assert stages.read_run_info(rda_out)["rda_model"] == "llama3.1:8b"
    assert stages.read_run_info(merged_out)["narrative_model"] == "gemma4:e4b"


def test_saved_json_from_the_same_model_is_reused(out, monkeypatch):
    monkeypatch.setattr(stages, "extract_rda", lambda *a, **k: pytest.fail("model called"))
    save_rda({"dmp": {"title": "cached"}}, "llama3.1:8b")
    strategy = FakeStrategy()
    _, merged_out = run(out, strategy)
    assert strategy.calls == 0
    assert json.loads(merged_out.read_text(encoding="utf-8"))["dmp"] == {"title": "cached"}


def test_saved_json_from_another_model_is_replaced(out, monkeypatch):
    monkeypatch.setattr(stages, "extract_rda", lambda text, model, host: ({"dmp": {"title": model}}, {}))
    save_rda({"dmp": {"title": "from llama3.1"}}, "llama3.1:8b")
    rda_out, _ = run(out, FakeStrategy(), rda_model="llama3.3:70b")
    assert json.loads(rda_out.read_text(encoding="utf-8")) == {"dmp": {"title": "llama3.3:70b"}}
    assert stages.read_run_info(rda_out)["rda_model"] == "llama3.3:70b"


def test_saved_json_from_an_earlier_wording_of_the_prompt_is_replaced(out, monkeypatch):
    monkeypatch.setattr(stages, "extract_rda", lambda *a, **k: ({"dmp": {"title": "new"}}, {}))
    save_rda({"dmp": {"title": "old wording"}}, "llama3.1:8b")
    record = json.loads(paths.run_info_path(paths.rda_path(3)).read_text(encoding="utf-8"))
    record["prompt_fingerprint"] = "000000000000"           # same version name, different text
    paths.run_info_path(paths.rda_path(3)).write_text(json.dumps(record), encoding="utf-8")
    rda_out, _ = run(out, FakeStrategy())
    assert json.loads(rda_out.read_text(encoding="utf-8")) == {"dmp": {"title": "new"}}


def test_saved_json_with_a_record_that_predates_the_fingerprint_is_replaced(out, monkeypatch):
    """Records written before the fingerprint existed lack the key entirely - that must
    count as a different run, not crash."""
    monkeypatch.setattr(stages, "extract_rda", lambda *a, **k: ({"dmp": {"title": "new"}}, {}))
    save_rda({"dmp": {"title": "old"}}, "llama3.1:8b")
    record = json.loads(paths.run_info_path(paths.rda_path(3)).read_text(encoding="utf-8"))
    del record["prompt_fingerprint"]
    paths.run_info_path(paths.rda_path(3)).write_text(json.dumps(record), encoding="utf-8")
    rda_out, _ = run(out, FakeStrategy())
    assert json.loads(rda_out.read_text(encoding="utf-8")) == {"dmp": {"title": "new"}}
    assert "prompt_fingerprint" in stages.read_run_info(rda_out)


def test_saved_json_without_a_run_record_is_replaced(out, monkeypatch):
    monkeypatch.setattr(stages, "extract_rda", lambda *a, **k: ({"dmp": {"title": "new"}}, {}))
    paths.rda_path(3).parent.mkdir(parents=True)
    paths.rda_path(3).write_text(json.dumps({"dmp": {"title": "unknown origin"}}), encoding="utf-8")
    rda_out, _ = run(out, FakeStrategy())
    assert json.loads(rda_out.read_text(encoding="utf-8")) == {"dmp": {"title": "new"}}


def test_invalid_json_saves_the_raw_answer_and_drops_the_old_model_output(out, monkeypatch):
    def fail(*a, **k):
        raise RdaExtractionError("no JSON", raw="{\"dmp\": {")
    monkeypatch.setattr(stages, "extract_rda", fail)
    save_rda({"dmp": {"title": "from llama3.1"}}, "llama3.1:8b")
    assert run(out, FakeStrategy(), rda_model="gemma4:e4b") == (None, None)
    assert paths.RDA_DIR.joinpath("sample3.raw.txt").read_text(encoding="utf-8") == "{\"dmp\": {"
    assert not paths.rda_path(3).exists()            # never left looking like gemma's answer


def test_no_narrative_json_means_no_merge(out, monkeypatch):
    monkeypatch.setattr(stages, "extract_rda", lambda *a, **k: ({"dmp": {}}, {}))
    rda_out, merged_out = stages.run_rda_stages(
        FakeStrategy(), out, 3, "llama3.1:8b", "other:model", "pdfplumber")
    assert rda_out.exists() and merged_out is None
