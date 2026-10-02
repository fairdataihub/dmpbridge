"""Score a model's RDA maDMP JSON against the hand-made reference for the same PDF.

Every field the model wrote is Correct or Hallucinated; every reference field the
model never got right is Missed. The rules mirror the manual Excel evaluation:

* identifiers (DOI, email, URL, licence) must be the same thing, ignoring
  ``https://``, ``www.``, ``doi.org/`` and case; a licence may be a name or a URL
* dates count when they are the same day, however they are written
* controlled values (``type``, ``language``, ``role`` ...) match ignoring case
* free text counts when at least CONTAINMENT_THRESHOLD of the model's words
  appear in the reference value (the project's own containment rule)
* empty is "", null, none, n/a, na, [] or {}

List items (datasets, contributors ...) are paired with the reference item of the
same title / name / identifier, otherwise with the item in the same position.
"""
import re

from dmpbridge.evaluation.evaluate import CONTAINMENT_THRESHOLD, containment, tokenize

EMPTY = {"", "null", "none", "n/a", "na"}
ONE_OR_MANY = {"contact_id", "contributor_id", "creator_id", "metadata_standard_id"}  # one object or a list
KEY_FIELDS = ("title", "name", "identifier", "license_ref")
ID_FIELDS = {"identifier", "mbox", "url", "access_url", "download_url", "scheme_uri", "license_ref"}
DATE_FIELDS = {"created", "modified", "issued", "start", "end", "start_date", "available_until"}
ENUM_FIELDS = {"type", "data_access", "personal_data", "sensitive_data", "ethical_issues_exist",
               "language", "relation_type", "resource_type", "funding_status", "role",
               "certified_with", "geo_location", "pid_system", "currency_code", "is_reused"}
LICENCE = [(re.compile(r"creativecommons\.org/licenses/([a-z\-]+)/(\d\.\d)"), r"cc-\1-\2"),
           (re.compile(r"^cc[\s\-]+([a-z][a-z\-]*)[\s\-]+(\d\.\d)$"), r"cc-\1-\2")]


# ── Documents as flat lists of fields ─────────────────────────────────────────

def is_empty(v):
    return v is None or v == [] or v == {} or (isinstance(v, str) and v.strip().lower() in EMPTY)


def canonical(node):
    """Wrap a single object into a one-item list wherever the schema allows either."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            v = canonical(v)
            if k in ONE_OR_MANY and isinstance(v, dict):
                v = [v]
            out[k] = v
        return out
    if isinstance(node, list):
        return [canonical(v) for v in node]
    return node


def flatten(node, path=""):
    """Every leaf of a document as (path, value)."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flatten(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from flatten(v, f"{path}[{i}]")
    else:
        yield path, node


def reference_fields(ref):
    """path -> value for every reference field that has a value."""
    return {p: v for p, v in flatten(canonical(ref)) if not is_empty(v)}


# ── Pairing list items ────────────────────────────────────────────────────────

def norm_text(v):
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def item_key(item):
    """The value that identifies a list item, if it has one."""
    if isinstance(item, dict):
        for k in KEY_FIELDS:
            if not is_empty(item.get(k)):
                return norm_text(item[k])
        return None
    return norm_text(item) if not is_empty(item) else None


def pair_items(model_list, ref_list):
    """model index -> reference index. Same title / name first, then same position."""
    ref_keys = {item_key(r): j for j, r in enumerate(ref_list)}
    ref_keys.pop(None, None)
    mapping, taken = {}, set()
    for i, m in enumerate(model_list):
        j = ref_keys.get(item_key(m))
        if j is not None and j not in taken:
            mapping[i], taken = j, taken | {j}
    for i, m in enumerate(model_list):
        if i in mapping:
            continue
        j = i if i not in taken else next((k for k in range(len(ref_list)) if k not in taken), None)
        if j is None:                                        # more items than the reference has
            j = len(ref_list) + len([x for x in mapping.values() if x >= len(ref_list)])
        mapping[i], taken = j, taken | {j}
    return mapping


def flatten_aligned(model, ref, path="", mpath=""):
    """Flatten the model document using the reference's list indices.
    Yields (reference-aligned path, the model's own path, value)."""
    if isinstance(model, dict):
        for k, v in model.items():
            r = ref.get(k) if isinstance(ref, dict) else None
            yield from flatten_aligned(v, r, f"{path}.{k}" if path else k, f"{mpath}.{k}" if mpath else k)
    elif isinstance(model, list):
        ref_list = ref if isinstance(ref, list) else []
        mapping = pair_items(model, ref_list)
        for i, v in enumerate(model):
            j = mapping[i]
            yield from flatten_aligned(v, ref_list[j] if j < len(ref_list) else None,
                                       f"{path}[{j}]", f"{mpath}[{i}]")
    else:
        yield path, mpath, model


# ── One field: right or wrong? ────────────────────────────────────────────────

def field_kind(path):
    last = re.sub(r"\[\d+\]$", "", path.split(".")[-1])
    if last in ID_FIELDS:
        return "identifier"
    if last in DATE_FIELDS:
        return "date"
    if last in ENUM_FIELDS:
        return "controlled"
    return "text"


def norm_id(v):
    s = norm_text(v)
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^(www\.|dx\.)?doi\.org/", "", s)
    s = re.sub(r"^www\.", "", s).rstrip("/")
    for pattern, repl in LICENCE:
        if pattern.search(s):
            return pattern.sub(repl, s)
    return s


def norm_date(v):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", norm_text(v))
    return f"{m[1]}-{m[2]}-{m[3]}" if m else norm_text(v)


def verdict(path, model_value, ref_value):
    """-> (\"Correct\" | \"Hallucinated\", reason)."""
    m_empty, r_empty = is_empty(model_value), is_empty(ref_value)
    if m_empty and r_empty:
        return "Correct", "both empty"
    if m_empty:
        return "Hallucinated", "empty where the reference has a value"
    if r_empty:
        return "Hallucinated", "not in the reference"
    kind = field_kind(path)
    if kind == "identifier":
        ok, how = norm_id(model_value) == norm_id(ref_value), "identifier differs"
    elif kind == "date":
        ok, how = norm_date(model_value) == norm_date(ref_value), "different day"
    elif kind == "controlled":
        ok, how = norm_text(model_value) == norm_text(ref_value), "different value"
    else:
        share = containment(tokenize(str(model_value)), tokenize(str(ref_value)))
        ok, how = share >= CONTAINMENT_THRESHOLD, f"only {share:.0%} of the model's words are in the reference"
    return ("Correct", "") if ok else ("Hallucinated", how)


# ── A whole document ──────────────────────────────────────────────────────────

def score(model_json, ref_json):
    """Every judged field of one model's output, as a list of dicts with keys
    field, model field, model value, reference value, verdict, reason."""
    ref = canonical(ref_json)
    ref_values = dict(flatten(ref))
    wanted = {p for p, v in ref_values.items() if not is_empty(v)}
    rows, covered = [], set()
    for path, mpath, mv in flatten_aligned(canonical(model_json), ref):
        rv = ref_values.get(path)
        v, why = verdict(path, mv, rv)
        if v == "Correct" and path in wanted:
            covered.add(path)
        rows.append({"field": path, "model field": mpath, "model value": mv,
                     "reference value": rv, "verdict": v, "reason": why})
    for path in sorted(wanted - covered):
        rows.append({"field": path, "model field": None, "model value": None,
                     "reference value": ref_values[path], "verdict": "Missed", "reason": ""})
    return rows


def metrics(rows, n_ref):
    """Counts and precision / recall / F1 for one model's rows."""
    c = sum(r["verdict"] == "Correct" for r in rows)
    h = sum(r["verdict"] == "Hallucinated" for r in rows)
    miss = sum(r["verdict"] == "Missed" for r in rows)
    out, found = c + h, n_ref - miss
    precision = c / out if out else 0.0
    recall = found / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"fields in reference": n_ref, "fields output": out, "correct": c, "hallucinated": h,
            "missed": miss, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
