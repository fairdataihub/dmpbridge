"""Score RDA maDMP JSON against a hand-made reference.

The rules are the ones the RDA notebooks use (notebooks/pdf-to-rda-dmp-json.ipynb and
notebooks/rda-field-matrix.ipynb), moved here unchanged so the pipeline and the
notebooks score the same way. Two views of one comparison:

* **Every field** (:func:`judge`, :func:`metrics`) - each field either side wrote is
  Correct, Hallucinated (a wrong value, a value the reference doesn't have, or blank
  where the reference has one) or Missed (the reference has it, the model never wrote
  it); precision, recall and F1 follow.
* **Annotated fields** (:func:`field_outcomes`) - only the fields the reference fills
  in, each Correct, Mismatch (a different value) or Missed (nothing there). This is the
  field matrix, drawn by :func:`matrix_figure`.

When are two values the same? Identifiers (DOI, email, URL, licence) must be the same
thing, ignoring ``https://`` and capitals; dates the same day; controlled values
(``type``, ``data_access``, ...) the same word; free text when at least 75% of the
model's words appear in the reference - the project's containment rule.
"""
import json
import re
from pathlib import Path

from ..evaluation.evaluate import CONTAINMENT_THRESHOLD, containment, tokenize

REFERENCE_DIR = Path(__file__).parent.parent.parent / "data" / "input" / "rda_ground_truth"
REFERENCE = "RDA_DMP_sample{n}_manual_annotation.json"

EMPTY       = {"", "null", "none", "n/a", "na"}                     # what counts as "nothing"
THRESHOLD   = CONTAINMENT_THRESHOLD                                 # share of the model's words in the reference
ONE_OR_MANY = {"contact_id", "contributor_id", "creator_id", "metadata_standard_id"}  # one object or a list

ID_FIELDS   = {"identifier", "mbox", "url", "access_url", "download_url", "scheme_uri", "license_ref"}
DATE_FIELDS = {"created", "modified", "issued", "start", "end", "start_date", "available_until"}
ENUM_FIELDS = {"type", "data_access", "personal_data", "sensitive_data", "ethical_issues_exist",
               "language", "relation_type", "resource_type", "funding_status", "role",
               "certified_with", "geo_location", "pid_system", "currency_code", "is_reused"}

VERDICTS = ["Correct", "Hallucinated", "Missed"]
OUTCOMES = ["Correct", "Mismatch", "Missed"]


def reference_path(sample: int, reference_dir: Path = REFERENCE_DIR) -> Path:
    """The hand-made reference for *sample*."""
    return Path(reference_dir) / REFERENCE.format(n=sample)


# ── Turning a document into fields ───────────────────────────────────────────

def is_empty(v) -> bool:
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


def load(path) -> dict:
    return canonical(json.loads(Path(path).read_text(encoding="utf-8")))


# ── Matching up list items (datasets, contributors, ...) ─────────────────────

KEY_FIELDS = ("title", "name", "identifier", "license_ref")


def norm_text(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def item_keys(item) -> list:
    """The values that identify a list item, in order of preference: its own title,
    name, identifier or licence, then the title of its host (a distribution is most
    recognisable by where it is kept - the annotator leaves distribution titles empty)."""
    if isinstance(item, dict):
        keys = [norm_text(item[k]) for k in KEY_FIELDS if not is_empty(item.get(k))]
        host = item.get("host")
        if isinstance(host, dict) and not is_empty(host.get("title")):
            keys.append(norm_text(host["title"]))
        return keys
    return [norm_text(item)] if not is_empty(item) else []


def item_key(item):
    """The first identifying value of a list item, if it has one."""
    keys = item_keys(item)
    return keys[0] if keys else None


def pair_items(model_list, ref_list) -> dict:
    """model index -> reference index. An item is paired with the reference item that
    shares any of its identifying values (title, name, identifier, licence, host title);
    what is left is paired by position. So a repository listed second by the model and
    first by the annotator still meets its counterpart."""
    ref_keys: dict = {}
    for j, r in enumerate(ref_list):
        for k in item_keys(r):
            ref_keys.setdefault(k, j)
    mapping, taken = {}, set()
    for i, m in enumerate(model_list):                       # pass 1: a shared identifying value
        for k in item_keys(m):
            j = ref_keys.get(k)
            if j is not None and j not in taken:
                mapping[i], taken = j, taken | {j}
                break
    for i, m in enumerate(model_list):                       # pass 1b: the same value, worded as the
        if i in mapping:                                     # verdict rule accepts it (75% of the
            continue                                         # model's words in the reference's)
        for k in item_keys(m):
            j = next((jj for rk, jj in ref_keys.items() if jj not in taken
                      and containment(tokenize(k), tokenize(rk)) >= THRESHOLD), None)
            if j is not None:
                mapping[i], taken = j, taken | {j}
                break
    for i, m in enumerate(model_list):                       # pass 2: same position, else a free slot
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


# ── Deciding whether a value is right ────────────────────────────────────────

def field_kind(path: str) -> str:
    last = re.sub(r"\[\d+\]$", "", path.split(".")[-1])
    if last in ID_FIELDS:
        return "identifier"
    if last in DATE_FIELDS:
        return "date"
    if last in ENUM_FIELDS:
        return "controlled"
    return "text"


LICENCE = [(re.compile(r"creativecommons\.org/licenses/([a-z\-]+)/(\d\.\d)"), r"cc-\1-\2"),
           (re.compile(r"^cc[\s\-]+([a-z][a-z\-]*)[\s\-]+(\d\.\d)$"), r"cc-\1-\2")]


def norm_id(v) -> str:
    s = norm_text(v)
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^(www\.|dx\.)?doi\.org/", "", s)
    s = re.sub(r"^www\.", "", s).rstrip("/")
    for pattern, repl in LICENCE:
        if pattern.search(s):
            return pattern.sub(repl, s)
    return s


def norm_date(v) -> str:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", norm_text(v))
    return f"{m[1]}-{m[2]}-{m[3]}" if m else norm_text(v)


def verdict(path: str, mv, rv) -> tuple[str, str]:
    """-> (verdict, reason)."""
    m_empty, r_empty = is_empty(mv), is_empty(rv)
    if m_empty and r_empty:
        return "Correct", "both empty"
    if m_empty:
        return "Hallucinated", "empty where the reference has a value"
    if r_empty:
        return "Hallucinated", "not in the reference"
    kind = field_kind(path)
    if kind == "identifier":
        ok, how = norm_id(mv) == norm_id(rv), "identifier differs"
    elif kind == "date":
        ok, how = norm_date(mv) == norm_date(rv), "different day"
    elif kind == "controlled":
        ok, how = norm_text(mv) == norm_text(rv), "different value"
    else:
        share = containment(tokenize(str(mv)), tokenize(str(rv)))
        ok, how = share >= THRESHOLD, f"only {share:.0%} of the model's words are in the reference"
    return ("Correct", "") if ok else ("Hallucinated", how)


# ── Scoring ──────────────────────────────────────────────────────────────────

def judge(model_doc: dict, ref_doc: dict) -> list[dict]:
    """One row per field: every field the model wrote, then every reference field it missed.

    Both documents as loaded by :func:`load` (``canonical`` applied)."""
    ref_values = dict(flatten(ref_doc))
    ref_fields = {p for p, v in ref_values.items() if not is_empty(v)}
    rows, covered = [], set()
    for path, mpath, mv in flatten_aligned(model_doc, ref_doc):
        rv = ref_values.get(path)
        v, why = verdict(path, mv, rv)
        if v == "Correct" and path in ref_fields:
            covered.add(path)
        rows.append({"field": path, "model field": mpath, "model value": mv,
                     "reference value": rv, "verdict": v, "reason": why})
    for path in sorted(ref_fields - covered):
        rows.append({"field": path, "model field": None, "model value": None,
                     "reference value": ref_values[path], "verdict": "Missed", "reason": ""})
    return rows


def n_reference_fields(ref_doc: dict) -> int:
    """How many fields the reference fills in."""
    return len({p for p, v in flatten(ref_doc) if not is_empty(v)})


def metrics(rows: list[dict], n_ref: int) -> dict:
    """Counts, precision, recall and F1 for the rows of :func:`judge`."""
    c = sum(r["verdict"] == "Correct" for r in rows)
    h = sum(r["verdict"] == "Hallucinated" for r in rows)
    miss = sum(r["verdict"] == "Missed" for r in rows)
    out, found = c + h, n_ref - miss
    precision = c / out if out else 0.0
    recall = found / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"fields in reference": n_ref, "fields output": out, "correct": c, "hallucinated": h,
            "missed": miss, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def _annotated(value) -> bool:
    """The reference holds a real value - the text "null", "none", "N/A" or "" does not count."""
    return value is not None and str(value).strip().lower() not in EMPTY


def field_outcomes(rows: list[dict]) -> dict[str, str]:
    """Annotated field -> Correct, Mismatch or Missed, from the rows of :func:`judge`.

    A field can have two rows (what the model wrote, and a Missed row); it is Correct if
    any row is, a Mismatch if the model wrote a value there, otherwise Missed."""
    by_field: dict[str, list[dict]] = {}
    for r in rows:
        if _annotated(r["reference value"]):
            by_field.setdefault(r["field"], []).append(r)
    outcomes = {}
    for field, rs in by_field.items():
        if any(r["verdict"] == "Correct" for r in rs):
            outcomes[field] = "Correct"
        elif any(r["verdict"] == "Hallucinated" and r["reason"] != "empty where the reference has a value"
                 for r in rs):
            outcomes[field] = "Mismatch"
        else:
            outcomes[field] = "Missed"
    return outcomes


def evaluate_file(model_path, ref_path) -> tuple[list[dict], dict, dict[str, str]]:
    """``(rows, metrics, field outcomes)`` for one model output against one reference."""
    ref = load(ref_path)
    rows = judge(load(model_path), ref)
    return rows, metrics(rows, n_reference_fields(ref)), field_outcomes(rows)


# ── The field matrix figure ──────────────────────────────────────────────────

COLOUR = {"Correct": "#238b45", "Mismatch": "#e34948", "Missed": "#898781"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

ROWS = [("Plan", "Plan title & description"), ("Plan", "Plan details (dates, ID, language, ethics)"),
        ("Contact", "Contact person"),
        ("Datasets", "Dataset title"), ("Datasets", "Dataset description"), ("Datasets", "Dataset type"),
        ("Datasets", "Personal & sensitive data"), ("Datasets", "Dataset identifier & dates"),
        ("Storage & licence", "Storage place (host)"), ("Storage & licence", "Data access"),
        ("Storage & licence", "Licence"), ("Storage & licence", "Other distribution details"),
        ("Metadata", "Metadata & standards"),
        ("People & project", "Contributors"), ("People & project", "Project & funding")]


def row_of(path: str) -> tuple[str, str]:
    """The (section, row) of the field matrix a field belongs to."""
    p = re.sub(r"\[\d+\]", "", path).replace("dmp.", "", 1)
    if p in ("title", "description"):                      return "Plan", "Plan title & description"
    if p.startswith("contact"):                            return "Contact", "Contact person"
    if p.startswith("contributor"):                        return "People & project", "Contributors"
    if p.startswith("project"):                            return "People & project", "Project & funding"
    if p.startswith("dataset.distribution.data_access"):   return "Storage & licence", "Data access"
    if p.startswith("dataset.distribution.host"):          return "Storage & licence", "Storage place (host)"
    if p.startswith("dataset.distribution.license"):       return "Storage & licence", "Licence"
    if p.startswith("dataset.distribution"):               return "Storage & licence", "Other distribution details"
    if p.startswith("dataset.metadata"):                   return "Metadata", "Metadata & standards"
    if p == "dataset.title":                               return "Datasets", "Dataset title"
    if p == "dataset.description":                         return "Datasets", "Dataset description"
    if p == "dataset.type":                                return "Datasets", "Dataset type"
    if p in ("dataset.personal_data", "dataset.sensitive_data"):
        return "Datasets", "Personal & sensitive data"
    if p.startswith("dataset"):                            return "Datasets", "Dataset identifier & dates"
    return "Plan", "Plan details (dates, ID, language, ethics)"


def matrix_figure(outcomes: dict[str, dict[str, str]], title: str, out_path=None, names=None):
    """The field matrix: rows = kinds of annotated field, columns = Correct / Mismatch /
    Missed per model, each cell the count and its share of the row; a Total row last.

    *outcomes* is ``{model: {field: outcome}}`` from :func:`field_outcomes`. Saves to
    *out_path* when given, and returns the matplotlib figure."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    names = names or {}
    models = list(outcomes)
    fields = sorted({f for o in outcomes.values() for f in o})
    n_fields = len(fields)
    n_row: dict = {}
    for f in fields:
        n_row[row_of(f)] = n_row.get(row_of(f), 0) + 1
    counts: dict = {}
    for m, o in outcomes.items():
        for f, outcome in o.items():
            key = (row_of(f), m, outcome)
            counts[key] = counts.get(key, 0) + 1
    totals = {m: {k: sum(v == k for v in outcomes[m].values()) for k in OUTCOMES} for m in models}

    TOTAL = ("Total", "Total")
    rows = [r for r in ROWS if r in n_row] + [TOTAL]
    n_row[TOTAL] = n_fields
    cols = []
    for k, m in enumerate(models):
        cols += [(m, o) for o in OUTCOMES] + ([None] if k < len(models) - 1 else [])

    def n_of(r, c):
        return totals[c[0]][c[1]] if r == TOTAL else counts.get((r, c[0], c[1]), 0)

    surface = np.array(mcolors.to_rgb(SURFACE))
    img = np.ones((len(rows), len(cols), 3)) * surface
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if c is None:
                continue
            n = n_of(r, c)
            strength = 0.12 + 0.88 * (n / n_row[r]) if n else 0.04
            img[i, j] = surface * (1 - strength) + np.array(mcolors.to_rgb(COLOUR[c[1]])) * strength

    fig, ax = plt.subplots(figsize=(max(6.5, 5.3 * len(models)), 0.52 * len(rows) + 2.6))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(img, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if c is None:
                continue
            n = n_of(r, c)
            share = n / n_row[r]
            if not n:
                ax.text(j, i, "–", ha="center", va="center", fontsize=13, color="#c9c8c3")
                continue
            ink = "white" if share > 0.5 else INK
            ax.text(j - 0.08, i, str(n), ha="right", va="center", fontsize=13, fontweight="bold", color=ink)
            ax.text(j + 0.02, i, f"{share:.0%}", ha="left", va="center", fontsize=10.5, color=ink)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[1]}  ({n_row[r]})" for r in rows], fontsize=12.5, color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")                # the Total row
    ax.tick_params(length=0)
    for i in range(len(rows) + 1):                                 # the same white gap between every pair of rows
        ax.axhline(i - 0.5, color=SURFACE, linewidth=3, zorder=3)
    for j in range(len(cols) + 1):                                 # and between every pair of columns
        ax.axvline(j - 0.5, color=SURFACE, linewidth=3, zorder=3)
    # Section names sit just left of the widest row label, measured, so they never overlap
    # it whatever the figure's width (one model or three).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
    axes_left = ax.get_window_extent(renderer).x0
    gap_points = (axes_left - label_left) * 72 / fig.dpi + 14
    for s in dict.fromkeys(r[0] for r in rows):
        idx = [i for i, r in enumerate(rows) if r[0] == s]
        if s != "Total":
            ax.annotate(s.upper(), xy=(0, (idx[0] + idx[-1]) / 2), xycoords=("axes fraction", "data"),
                        xytext=(-gap_points, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=10, fontweight="bold", color=MUTED)
        if idx[0] > 0:                                             # a wider white gap between sections
            ax.axhline(idx[0] - 0.5, color=SURFACE, linewidth=14 if s == "Total" else 9, zorder=4)

    ax.set_xticks([j for j, c in enumerate(cols) if c])
    ax.set_xticklabels([c[1] for c in cols if c], fontsize=11.5, color=INK)
    ax.xaxis.tick_top()
    for m in models:
        first = cols.index((m, OUTCOMES[0]))
        t = totals[m]
        ax.annotate(names.get(m, m), xy=(first + 1, 1), xycoords=("data", "axes fraction"), xytext=(0, 38),
                    textcoords="offset points", ha="center", va="bottom", fontsize=15, fontweight="bold",
                    color=INK)
        ax.annotate(f"{t['Correct']} correct · {t['Mismatch']} mismatch · {t['Missed']} missed",
                    xy=(first + 1, 1), xycoords=("data", "axes fraction"), xytext=(0, 22),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10.5, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=78, color=INK)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return fig
