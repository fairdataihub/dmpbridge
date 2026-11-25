"""Checks that a result has the shape the web app reads.

The web app's review steps can fix anything inside this shape: wrong labels, too many
sections, a missing dataset. What they cannot take is a different shape, so a result
that fails here is run again rather than sent.

Each check returns ``None`` when the document fits, or what is wrong with it.
"""
from typing import Any, Optional


def _str(value: Any) -> bool:
    return isinstance(value, str)


def _list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, dict) for v in value)


def check_narrative(doc: Any) -> Optional[str]:
    """``{"narrative": {"template": {"title", "section": [{"title", "description", "order",
    "question": [{"text", "order", "answer": {"json": {"answer"}}}]}]}}}``, with at
    least one section."""
    template = doc.get("narrative", {}).get("template") if isinstance(doc, dict) else None
    if not isinstance(template, dict) or not _str(template.get("title")):
        return "no template title"
    sections = template.get("section")
    if not _list_of_dicts(sections) or not sections:
        return "no sections"
    for s, section in enumerate(sections, 1):
        if not (_str(section.get("title")) and _str(section.get("description", ""))
                and isinstance(section.get("order"), int) and _list_of_dicts(section.get("question"))):
            return f"section {s} is malformed"
        for q, question in enumerate(section["question"], 1):
            answer = question.get("answer", {}).get("json", {}) if isinstance(question.get("answer"), dict) else None
            if not (_str(question.get("text")) and isinstance(question.get("order"), int)
                    and isinstance(answer, dict) and _str(answer.get("answer"))):
                return f"question {s}.{q} is malformed"
    return None


def check_rda(doc: Any) -> Optional[str]:
    """``{"dmp": {...}}`` where every part the metadata step edits has its expected type:
    ``dataset`` a list of objects, and in each dataset ``dataset_id`` an object and
    ``distribution``, ``metadata``, ``creator`` lists of objects."""
    dmp = doc.get("dmp") if isinstance(doc, dict) else None
    if not isinstance(dmp, dict):
        return "no dmp object"
    datasets = dmp.get("dataset", [])
    if not _list_of_dicts(datasets):
        return "dataset is not a list of objects"
    for d, dataset in enumerate(datasets, 1):
        if not isinstance(dataset.get("dataset_id", {}), dict):
            return f"dataset {d}: dataset_id is not an object"
        for key in ("distribution", "metadata", "creator"):
            if not _list_of_dicts(dataset.get(key, [])):
                return f"dataset {d}: {key} is not a list of objects"
        for distribution in dataset.get("distribution", []):
            if not isinstance(distribution.get("host", {}), dict):
                return f"dataset {d}: host is not an object"
            if not _list_of_dicts(distribution.get("license", [])):
                return f"dataset {d}: license is not a list of objects"
    return None
