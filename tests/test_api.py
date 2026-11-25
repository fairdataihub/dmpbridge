"""The HTTP API (dmpbridge.api).

No Ollama server: the pipeline is replaced by a fake runner, and for the runner itself
the extraction and the model calls are faked.
"""
import json
import random
import threading
import time
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from dmpbridge.api import app as api
from dmpbridge.api import pipeline
from dmpbridge.api.jobs import JobFailed, JobQueue
from dmpbridge.api.shape import check_narrative, check_rda
from dmpbridge.core.converter import to_structured
from dmpbridge.evaluation.annotation_rules import apply_new_annotation_rules
from dmpbridge.extractors.fallback import ExtractionResult
from dmpbridge.rda import RdaExtractionError

PDF = b"%PDF-1.7 fake"
MERGED = {"dmp": {"title": "T"}, "narrative": {"template": {"title": "T"}}}


def client_with(run, monkeypatch, **queue_args):
    monkeypatch.setattr(api, "jobs", JobQueue(run=run, **queue_args))
    return TestClient(api.app)


def upload(client, data=PDF):
    return client.post("/jobs", files={"file": ("plan.pdf", data, "application/pdf")})


def wait(client, job_id, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_runs_a_job_and_returns_the_result(monkeypatch):
    seen = {}

    def run(pdf_path, set_stage):
        seen["pdf"] = pdf_path
        seen["bytes"] = pdf_path.read_bytes()
        set_stage("narrative")
        return MERGED

    with client_with(run, monkeypatch) as client:
        res = upload(client)
        assert res.status_code == 202
        body = wait(client, res.json()["id"])

    assert body["status"] == "done"
    assert body["result"] == MERGED
    assert body["stage"] is None
    assert seen["bytes"] == PDF
    assert not seen["pdf"].parent.exists()      # the upload is deleted after the run


def test_each_upload_gets_its_own_folder(monkeypatch):
    folders = []

    def run(pdf_path, set_stage):
        folders.append(pdf_path.parent)
        return MERGED

    with client_with(run, monkeypatch) as client:
        for _ in range(2):
            wait(client, upload(client).json()["id"])

    assert len(set(folders)) == 2


@pytest.mark.parametrize("data, status", [(b"not a pdf", 415), (PDF + b"x" * 20, 413)])
def test_rejects_bad_uploads(monkeypatch, data, status):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", len(PDF) + 10)
    with client_with(lambda *_: MERGED, monkeypatch) as client:
        assert upload(client, data).status_code == status


def test_shows_a_job_failure_to_the_user(monkeypatch):
    def run(pdf_path, set_stage):
        raise JobFailed("The text in this PDF can't be read.")

    with client_with(run, monkeypatch) as client:
        body = wait(client, upload(client).json()["id"])

    assert body["status"] == "failed"
    assert body["error"] == "The text in this PDF can't be read."


def test_hides_unexpected_errors(monkeypatch):
    def run(pdf_path, set_stage):
        raise KeyError("internal detail")

    with client_with(run, monkeypatch) as client:
        body = wait(client, upload(client).json()["id"])

    assert body["status"] == "failed"
    assert "internal detail" not in body["error"]


def test_unknown_job_is_404(monkeypatch):
    with client_with(lambda *_: MERGED, monkeypatch) as client:
        assert client.get("/jobs/nope").status_code == 404


def test_finished_jobs_expire(monkeypatch):
    with client_with(lambda *_: MERGED, monkeypatch, ttl=0.5) as client:
        job_id = upload(client).json()["id"]
        wait(client, job_id)
        time.sleep(0.6)
        assert client.get(f"/jobs/{job_id}").status_code == 404


def test_reports_queue_position_and_limit(monkeypatch):
    release = threading.Event()

    def run(pdf_path, set_stage):
        release.wait(5)
        return MERGED

    with client_with(run, monkeypatch, max_waiting=1) as client:
        first = upload(client).json()["id"]
        while client.get(f"/jobs/{first}").json()["status"] != "running":
            time.sleep(0.01)
        second = upload(client).json()["id"]
        assert client.get(f"/jobs/{second}").json()["position"] == 1
        assert upload(client).status_code == 503     # one already waiting
        release.set()
        assert wait(client, second)["status"] == "done"


# ── The runner: the pipeline's stages, faked ──────────────────────────────────

GOOD_BLOCKS = [{"text": "Plan", "label": "title"}, {"text": "Data", "label": "section.title"},
               {"text": "What data?", "label": "question.text"}, {"text": "Images.", "label": "answer.text"}]
GOOD_RDA = {"dmp": {"title": "Plan", "dataset": [{"title": "Images", "distribution": [{"host": {}}]}]}}


@pytest.fixture
def fake_stages(monkeypatch):
    """Each model call takes the next answer from ``calls["blocks"]`` / ``calls["rda"]``."""
    calls = {"blocks": [GOOD_BLOCKS], "rda": [GOOD_RDA]}

    class FakeStrategy:
        def __init__(self, **kwargs):
            pass

        def classify_entire_document(self, text):
            return calls["blocks"].pop(0) if len(calls["blocks"]) > 1 else calls["blocks"][0]

    def extract(pdf_path, primary, fallbacks, **kwargs):
        calls["cache_write"] = kwargs.get("cache_write")
        calls["cache_lookup"] = kwargs.get("cache_lookup")
        return calls.get("read", ExtractionResult("Plan text", "pdfplumber", True))

    def rda(text, model, host):
        answer = calls["rda"].pop(0) if len(calls["rda"]) > 1 else calls["rda"][0]
        if isinstance(answer, Exception):
            raise answer
        return answer, {}

    monkeypatch.setattr(pipeline, "extract_with_fallback", extract)
    monkeypatch.setattr(pipeline, "WholeDocStrategy", FakeStrategy)
    monkeypatch.setattr(pipeline, "extract_rda", rda)
    return calls


def run_pipeline(tmp_path, stages=None):
    run = pipeline.make_runner("gemma4:e4b", "gemma4:e4b", "http://ollama")
    return run(tmp_path / "upload.pdf", (stages if stages is not None else []).append)


def test_runner_merges_both_parts_without_caching(fake_stages, tmp_path):
    stages = []
    result = run_pipeline(tmp_path, stages)

    assert stages == ["reading", "narrative", "rda"]
    assert list(result) == ["dmp", "narrative"]
    assert fake_stages["cache_lookup"] is None
    assert fake_stages["cache_write"] is not None      # replaces the default, which writes to data/output


def test_runner_rejects_unreadable_text(fake_stages, tmp_path):
    fake_stages["read"] = ExtractionResult("@@##", "pdfplumber", False, reason="garbled")
    with pytest.raises(JobFailed, match="can't be read"):
        run_pipeline(tmp_path)


def test_runner_retries_an_empty_narrative(fake_stages, tmp_path):
    fake_stages["blocks"] = [[], GOOD_BLOCKS]          # e.g. the answer was not valid JSON
    result = run_pipeline(tmp_path)
    assert result["narrative"]["template"]["section"][0]["title"] == "Data"


@pytest.mark.parametrize("bad", [
    RdaExtractionError("not valid JSON", raw="{"),
    {"dmp": {"dataset": {"title": "not a list"}}},
    {"no": "dmp"},
])
def test_runner_retries_a_bad_rda_answer(fake_stages, tmp_path, bad):
    fake_stages["rda"] = [bad, GOOD_RDA]
    assert run_pipeline(tmp_path)["dmp"] == GOOD_RDA["dmp"]


def test_runner_gives_up_after_three_tries(fake_stages, tmp_path):
    fake_stages["rda"] = [RdaExtractionError("not valid JSON", raw="{")]
    with pytest.raises(JobFailed, match="metadata couldn't be extracted"):
        run_pipeline(tmp_path)


# ── Whatever the model labels, the structure fits the web app ─────────────────

LABELS = ["title", "section.title", "section.description", "question.text", "answer.text"]


def test_any_labeling_fits_the_web_app():
    """The model only chooses a label for each block; any order of labels must still
    give the section / question shape. 20,000 random labelings, including empty text."""
    rng = random.Random(0)
    for _ in range(20_000):
        blocks = [{"text": rng.choice(["", "  ", "Text", "More text"]), "label": rng.choice(LABELS)}
                  for _ in range(rng.randint(0, 25))]
        doc = apply_new_annotation_rules(to_structured(blocks))
        has_body = any(b["text"].strip() and b["label"] != "title" for b in blocks)
        assert check_narrative(doc) == (None if has_body else "no sections"), blocks


def saved_outputs(stage):
    root = Path(__file__).parent.parent / "data" / "output" / stage
    return sorted(p for p in root.rglob("sample*.json") if not p.name.endswith(".run.json"))


def test_every_saved_narrative_fits():
    """Every narrative the pipeline has saved, for every model and sample."""
    files = saved_outputs("4_final")
    assert files
    for path in files:
        assert check_narrative(json.loads(path.read_text(encoding="utf-8"))) is None, path


def test_every_saved_rda_fits():
    files = saved_outputs("5_rda")
    assert files
    for path in files:
        assert check_rda(json.loads(path.read_text(encoding="utf-8"))) is None, path


def test_runner_retries_a_timed_out_answer(fake_stages, tmp_path):
    fake_stages["rda"] = [requests.exceptions.ReadTimeout("read timed out"), GOOD_RDA]
    assert run_pipeline(tmp_path)["dmp"] == GOOD_RDA["dmp"]


def test_model_calls_use_the_api_timeout(monkeypatch):
    """The API's timeout reaches the request; the command line keeps its hour."""
    from dmpbridge.models.ollama import OllamaModel

    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "[]"}

    monkeypatch.setattr(requests, "get", lambda *a, **k: Response())
    monkeypatch.setattr(requests, "post", lambda *a, **k: seen.update(k) or Response())
    OllamaModel("gemma4:e4b", "http://ollama").complete("s", "p", schema={})
    assert seen["timeout"] == api.TIMEOUT_SECONDS == 180
