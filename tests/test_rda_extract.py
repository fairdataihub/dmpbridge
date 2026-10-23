"""dmpbridge.rda: the RDA extraction step (stage 5) and the merge (stage 6).

No Ollama server: the model backend is replaced by a fake that returns a fixed answer.
"""
import json

import pytest

from dmpbridge.core import paths
from datetime import date

from dmpbridge.rda import (RdaExtractionError, extract_rda, load_output_schema, load_schema, make_prompt,
                           merge, parse_prompt_file, prompt_fingerprint, system_prompt)
from dmpbridge.rda import extract as rda_extract


class FakeModel:
    answer = "{}"
    done_reason = "stop"
    seen = {}

    def __init__(self, **kwargs):
        FakeModel.seen = dict(kwargs)
        self.last_call = None

    def complete(self, system, prompt, *, schema):
        FakeModel.seen.update(system=system, prompt=prompt, schema=schema)
        self.last_call = {"eval_count": 7, "done_reason": FakeModel.done_reason}
        return FakeModel.answer


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(rda_extract, "OllamaModel", FakeModel)
    FakeModel.answer, FakeModel.done_reason = "{}", "stop"
    return FakeModel


def test_schema_has_no_refs_left():
    schema, text = load_schema()
    assert "$ref" not in text and "$defs" not in text
    assert "dmp" in schema["properties"]


def test_prompt_holds_the_plan_text_but_not_the_schema():
    prompt = make_prompt("PLAN TEXT HERE")
    assert "PLAN TEXT HERE" in prompt
    assert load_schema()[1] not in prompt          # since v10c the schema is the output constraint only


def test_prompt_file_splits_into_system_and_user_parts():
    """The package's prompt.txt parses; both parts are what the file says, byte for byte."""
    text = rda_extract.PROMPT_FILE.read_text(encoding="utf-8")
    system, user = parse_prompt_file(text)
    assert text.startswith("SYSTEM:\n" + system.rstrip("\n"))
    assert text.rstrip("\n").endswith(user)
    assert "{dmp_text}" in user and "{dmp_text}" not in system
    assert system_prompt() == system.replace("<<TODAY>>", date.today().isoformat())


def test_parse_prompt_file_accepts_the_notebooks_saved_header():
    system, user = parse_prompt_file(
        "SYSTEM:\nRules.\n\n\nPROMPT (the schema and the document text go in the braces):\nPlan: {dmp_text}\n")
    assert (system, user) == ("Rules.\n", "Plan: {dmp_text}")


@pytest.mark.parametrize("text", ["SYSTEM:\nRules.\nno separator {dmp_text}", "SYSTEM:\nRules.\n\nPROMPT:\nno placeholder"])
def test_parse_prompt_file_rejects_a_broken_file(text):
    with pytest.raises(ValueError):
        parse_prompt_file(text)


def test_today_placeholder_is_filled_in_but_not_fingerprinted(monkeypatch, tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("SYSTEM:\nDefault date <<TODAY>>.\n\nPROMPT:\n{dmp_text}\n", encoding="utf-8")
    monkeypatch.setattr(rda_extract, "PROMPT_FILE", p)
    rda_extract._prompt_parts.cache_clear()
    try:
        assert system_prompt() == f"Default date {date.today().isoformat()}.\n"
        assert "<<TODAY>>" not in system_prompt()
        fp = prompt_fingerprint()
        assert len(fp) == 12 and fp == prompt_fingerprint()      # stable; the date is not in it
    finally:
        rda_extract._prompt_parts.cache_clear()


def test_extract_rda_sends_the_evaluated_settings(fake):
    fake.answer = '{"dmp": {"title": "A plan"}}'
    rda, info = extract_rda("plan", "llama3.1:8b")
    assert rda == {"dmp": {"title": "A plan"}}
    assert info["eval_count"] == 7
    assert fake.seen["model"] == "llama3.1:8b"
    assert fake.seen["num_ctx"] == 32768 and fake.seen["num_predict"] == 8000
    assert fake.seen["options"] == {"repeat_last_n": 1024}
    assert fake.seen["schema"] == load_output_schema()          # the constraint, not the full standard


def test_output_schema_is_the_evaluated_constraint():
    """The package copy must be the schema the notebook run was evaluated with, byte for
    byte, and it must make the parts the smaller models used to skip mandatory."""
    from pathlib import Path
    evaluated = Path(f"scripts/build/prompts/rda_schema_{rda_extract.PROMPT_VERSION}.json")
    if evaluated.exists():
        assert rda_extract.OUTPUT_SCHEMA_FILE.read_bytes() == evaluated.read_bytes()
    dataset = load_output_schema()["properties"]["dmp"]["properties"]["dataset"]
    assert dataset["minItems"] == 1
    for part in ("description", "distribution", "metadata"):
        assert part in dataset["items"]["required"]
    assert dataset["items"]["properties"]["distribution"]["minItems"] == 1


def test_fingerprint_changes_with_the_output_schema(monkeypatch, tmp_path):
    before = prompt_fingerprint()
    p = tmp_path / "output_schema.json"
    p.write_text('{"type": "object", "properties": {"dmp": {"type": "object"}}}', encoding="utf-8")
    monkeypatch.setattr(rda_extract, "OUTPUT_SCHEMA_FILE", p)
    rda_extract.load_output_schema.cache_clear()
    try:
        assert prompt_fingerprint() != before
    finally:
        rda_extract.load_output_schema.cache_clear()


def test_invalid_json_raises_with_the_raw_answer(fake):
    fake.answer, fake.done_reason = '{"dmp": {"dataset": [{"title": "x"', "length"
    with pytest.raises(RdaExtractionError, match="token cap") as e:
        extract_rda("plan", "llama3.1:8b")
    assert e.value.raw == fake.answer


def test_merge_puts_rda_first_then_narrative():
    merged = merge({"narrative": {"a": 1}}, {"dmp": {"b": 2}})
    assert merged == {"dmp": {"b": 2}, "narrative": {"a": 1}}
    assert list(merged) == ["dmp", "narrative"]


def test_merge_refuses_overlapping_keys():
    with pytest.raises(ValueError):
        merge({"dmp": {}}, {"dmp": {}})


def test_stage_paths():
    assert paths.rda_path(3) == paths.OUTPUT_ROOT / "5_rda" / "sample3.json"
    assert paths.merged_path(3) == paths.OUTPUT_ROOT / "6_merged" / "sample3.json"
    assert paths.run_info_path(paths.rda_path(3)) == paths.OUTPUT_ROOT / "5_rda" / "sample3.run.json"
