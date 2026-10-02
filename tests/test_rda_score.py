"""Tests for dmpbridge.rda.score: the field-by-field rules of the RDA maDMP
evaluation, mirroring the manual Excel workbook, and the pairing of list items."""
from dmpbridge.rda.score import metrics, pair_items, reference_fields, score, verdict


# ── One field ────────────────────────────────────────────────────────────────

def test_identifiers_ignore_scheme_and_doi_prefix():
    assert verdict("dmp.dataset[0].dataset_id.identifier", "https://doi.org/10.1/X", "10.1/x")[0] == "Correct"
    assert verdict("dmp.dataset[0].dataset_id.identifier", "10.1/X", "10.2/Y")[0] == "Hallucinated"


def test_licence_name_equals_licence_url():
    path = "dmp.dataset[0].distribution[0].license[0].license_ref"
    assert verdict(path, "https://creativecommons.org/licenses/by/4.0/", "CC BY 4.0")[0] == "Correct"


def test_dates_compare_by_day():
    assert verdict("dmp.project[0].start", "2015-05-12T00:00:00Z", "2015-05-12")[0] == "Correct"
    assert verdict("dmp.project[0].start", "2015-05-13", "2015-05-12")[0] == "Hallucinated"


def test_free_text_uses_containment_of_the_models_words():
    assert verdict("dmp.dataset[0].title", "Hakai JSP Time Series", "Hakai JSP Time Series Data")[0] == "Correct"
    assert verdict("dmp.title", "Hakai JSP Time Series Data Management Plan", "Hakai JSP Time Series")[0] == "Hallucinated"


def test_empty_values():
    assert verdict("dmp.contact.mbox", "N/A", None) == ("Correct", "both empty")
    assert verdict("dmp.contact.mbox", "a@b.org", None)[0] == "Hallucinated"
    assert verdict("dmp.contact.name", "", "Brett Johnson")[0] == "Hallucinated"


# ── Lists ────────────────────────────────────────────────────────────────────

def test_items_pair_by_title_then_by_position():
    model = [{"title": "B"}, {"title": "A"}, {"title": "new"}]
    ref = [{"title": "A"}, {"title": "B"}]
    assert pair_items(model, ref) == {0: 1, 1: 0, 2: 2}


# ── A whole document ─────────────────────────────────────────────────────────

REF = {"dmp": {"title": "Plan", "contact": {"name": "Ann", "mbox": None, "contact_id": {"identifier": "0000-1", "type": "orcid"}},
               "dataset": [{"title": "D1", "type": "dataset"}, {"title": "D2"}]}}


def test_score_counts_correct_hallucinated_and_missed():
    out = {"dmp": {"title": "Plan", "contact": {"name": "Bob", "contact_id": [{"identifier": "0000-1", "type": "orcid"}]},
                   "dataset": [{"title": "D2"}]}}
    rows = score(out, REF)
    by = {(r["field"], r["verdict"]) for r in rows}
    assert ("dmp.title", "Correct") in by
    assert ("dmp.contact.name", "Hallucinated") in by and ("dmp.contact.name", "Missed") in by
    assert ("dmp.dataset[1].title", "Correct") in by                 # paired with the reference's D2 by title
    assert ("dmp.dataset[0].title", "Missed") in by and ("dmp.dataset[0].type", "Missed") in by
    m = metrics(rows, len(reference_fields(REF)))
    assert (m["correct"], m["hallucinated"], m["missed"]) == (4, 1, 3)
    assert m["fields in reference"] == 7 and m["precision"] == 0.8


def test_single_object_or_list_are_equivalent():
    out = {"dmp": {"contact": {"contact_id": {"identifier": "0000-1", "type": "orcid"}}}}
    rows = score(out, REF)
    assert all(r["verdict"] == "Correct" for r in rows if r["model field"] is not None)
