"""dmpbridge.rda.evaluate: the RDA scoring rules, and the dmpbridge-evaluate-rda command.

The rules were moved unchanged from the RDA notebooks; the last test checks they still
give the notebooks' saved result for sample 3.
"""
import json
from pathlib import Path

import pytest

from dmpbridge.core import paths
from dmpbridge.rda import evaluate as E

REF = {"dmp": {"title": "Cyber physical systems plan", "language": "eng",
               "contact": {"mbox": "a@b.org", "contact_id": {"identifier": "0000-0001", "type": "orcid"}},
               "dataset": [{"title": "Field data", "personal_data": "no"},
                           {"title": "Simulation code", "personal_data": "no"}]}}


def score(model_doc):
    rows = E.judge(E.canonical(model_doc), E.canonical(REF))
    return rows, E.field_outcomes(rows)


@pytest.mark.parametrize("path, mv, rv, want", [
    ("dmp.title", "Cyber physical systems", "Cyber physical systems plan", "Correct"),     # 100% of words in ref
    ("dmp.title", "Cyber physical systems plan for 2015", "Cyber physical systems plan", "Hallucinated"),
    ("dmp.contact.mbox", "A@B.org", "a@b.org", "Correct"),                                 # identifiers: case
    ("x.license_ref", "https://creativecommons.org/licenses/by/4.0/", "CC BY 4.0", "Correct"),
    ("dmp.created", "2015-05-12T00:00:00Z", "2015-05-12", "Correct"),                     # dates: same day
    ("dmp.language", "en", "eng", "Hallucinated"),                                         # controlled: exact
    ("dmp.contact.name", "N/A", "Brett", "Hallucinated"),                                  # empty vs a value
    ("dmp.contact.name", "", None, "Correct"),                                             # both empty
])
def test_verdict(path, mv, rv, want):
    assert E.verdict(path, mv, rv)[0] == want


def test_datasets_are_paired_by_title_not_position():
    model = {"dmp": {"dataset": [{"title": "Simulation code"}, {"title": "Field data"}]}}
    _, outcomes = score(model)
    assert outcomes["dmp.dataset[0].title"] == "Correct"
    assert outcomes["dmp.dataset[1].title"] == "Correct"


def test_field_matrix_outcomes():
    model = {"dmp": {"title": "Cyber physical systems plan", "language": "en",
                     "contact": {"mbox": "", "contact_id": [{"identifier": "0000-0001", "type": "orcid"}]}}}
    rows, outcomes = score(model)
    assert outcomes["dmp.title"] == "Correct"
    assert outcomes["dmp.language"] == "Mismatch"                 # wrote a different value
    assert outcomes["dmp.contact.mbox"] == "Missed"               # wrote it empty
    assert outcomes["dmp.dataset[0].title"] == "Missed"           # never wrote it
    assert outcomes["dmp.contact.contact_id[0].identifier"] == "Correct"   # one object == a list of one
    assert len(outcomes) == E.n_reference_fields(E.canonical(REF)) == 9


def test_metrics():
    m = E.metrics([{"verdict": "Correct"}] * 3 + [{"verdict": "Hallucinated"}] + [{"verdict": "Missed"}] * 2, 5)
    assert (m["correct"], m["hallucinated"], m["missed"], m["fields output"]) == (3, 1, 2, 4)
    assert (m["precision"], m["recall"], m["f1"]) == (0.75, 0.6, 0.667)


def test_evaluate_rda_writes_matrix_and_spreadsheet(tmp_path, monkeypatch):
    from dmpbridge.cli.rda_evaluate_cmd import evaluate_rda
    monkeypatch.setattr(paths, "RDA_DIR", tmp_path / "5_rda")
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / E.REFERENCE.format(n=3)).write_text(json.dumps(REF), encoding="utf-8")
    out = paths.rda_path(3)
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"dmp": {"title": "Cyber physical systems plan"}}), encoding="utf-8")
    paths.run_info_path(out).write_text(json.dumps({"rda_model": "m:1"}), encoding="utf-8")
    pooled = evaluate_rda(reference_dir=refs)
    assert pooled["correct"] == 1 and pooled["missed"] == 8
    assert (paths.RDA_DIR / "evaluation" / "matrix_sample3.png").exists()
    assert (paths.RDA_DIR / "evaluation" / "evaluation_results.xlsx").exists()


RUN = Path("notebooks/rda-experiments/runs/v10-narrative")


@pytest.mark.skipif(not RUN.exists() or not E.reference_path(3).exists(), reason="saved run not on disk")
@pytest.mark.parametrize("model, want", [("llama3.1-8b", (10, 0, 15)), ("gemma4-e4b", (10, 7, 8)),
                                         ("llama3.3-70b", (12, 4, 9))])
def test_matches_the_notebooks_saved_result(model, want):
    _, _, outcomes = E.evaluate_file(RUN / f"sample3.rda.{model}.json", E.reference_path(3))
    assert tuple(sum(v == k for v in outcomes.values()) for k in E.OUTCOMES) == want
