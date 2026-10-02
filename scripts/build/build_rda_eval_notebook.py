"""Build notebooks/evaluate-rda-json.ipynb.

Scores each model's RDA maDMP JSON against the hand-made reference JSON for the
same sample: every field a model wrote is Correct or Hallucinated, every
reference field it never got right is Missed, and precision / recall / F1 follow.

    python scripts/build/build_rda_eval_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/evaluate-rda-json.ipynb")


def md(cid, body):
    """Markdown cell from a plain block of text."""
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in body.strip("\n").split("\n")]}


def code(cid, body):
    """Code cell from a plain block of text."""
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in body.strip("\n").split("\n")]}


cells = [

md("title", '''
# Evaluating the RDA DMP JSON outputs

Three models turned a DMP PDF into RDA maDMP JSON. This notebook scores each model's
JSON against the manually written reference JSON for the same sample and reports how
much each model got right, got wrong, and left out.

**Every field gets one of three verdicts**

| Verdict | Meaning |
|---|---|
| **Correct** | the model wrote a field and its value matches the reference — or both are empty |
| **Hallucinated** | the model wrote a field whose value is wrong, invented (the reference has nothing there), or empty where the reference has a value |
| **Missed** | a reference field the model never produced correctly |

Every field the model wrote is Correct or Hallucinated, so `Correct + Hallucinated` is
exactly "fields the model output". Missed is counted on the reference side.

**Three numbers summarise a model**

- **precision** = correct ÷ fields output — of what the model said, how much was right
- **recall** = fields it got right ÷ fields in the reference — of what the document contains, how much was found
- **F1** — the balance of the two (section 5 shows the arithmetic with this run's numbers)

**When do two values count as the same?** Strict for identifiers, lenient for prose:

| Kind of field | Rule | Passes |
|---|---|---|
| identifiers, emails, URLs | exact, ignoring `https://`, `www.`, `doi.org/`, case and a trailing `/` | `https://doi.org/10.21966/1.566666` = `10.21966/1.566666` |
| licences | exact, after mapping to a short form | `https://creativecommons.org/licenses/by/4.0/` = `CC BY 4.0` |
| dates | same calendar day | `2015-05-12T00:00:00Z` = `2015-05-12` |
| controlled values (`type`, `data_access`, yes/no fields, `language`, roles) | exact, case-insensitive | `DOI` = `doi` |
| free text (titles, names, descriptions) | at least 75% of the model's words appear in the reference value — the project's usual containment rule | `Hakai JSP Time Series` ≈ `Hakai Institute Juvenile Salmon Program Time Series` |

Saying *less* than the reference can still match; saying *more* cannot — a title with
"Data Management Plan" appended fails, because only 7 of its 10 words are in the reference.

**Sections:** 1 flatten · 2 line up list items · 3 judge · 4 score · 5 metrics ·
6 detail tables · 7 where the misses are · 8 charts · 9 save
'''),

code("setup", r'''
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

# The project's own containment rule, so this evaluation and Path A / B agree.
from dmpbridge.evaluation.evaluate import tokenize, containment, CONTAINMENT_THRESHOLD

# ── Where things are ─────────────────────────────────────────────────────────
RDA_DIR   = Path("data/output/rda")
MODELS    = ["llama3.1-8b", "gemma4-e4b", "llama3.3-70b"]      # as in the file names
REFERENCE = "RDA_DMP_sample{n}_manual_annotation.json"           # one per sample
OUTPUT    = "sample{n}.rda.{model}.json"                         # one per sample and model
RESULTS   = RDA_DIR / "evaluation_results.xlsx"
CHART     = RDA_DIR / "evaluation_charts.png"

# ── Judging rules ────────────────────────────────────────────────────────────
EMPTY       = {"", "null", "none", "n/a", "na"}                     # what counts as "nothing"
THRESHOLD   = CONTAINMENT_THRESHOLD                                 # 0.75: share of the model's words that must be in the reference
ONE_OR_MANY = {"contact_id", "contributor_id", "creator_id", "metadata_standard_id"}  # schema allows one object or a list

ID_FIELDS   = {"identifier", "mbox", "url", "access_url", "download_url", "scheme_uri", "license_ref"}
DATE_FIELDS = {"created", "modified", "issued", "start", "end", "start_date", "available_until"}
ENUM_FIELDS = {"type", "data_access", "personal_data", "sensitive_data", "ethical_issues_exist",
               "language", "relation_type", "resource_type", "funding_status", "role",
               "certified_with", "geo_location", "pid_system", "currency_code", "is_reused"}

# ── Chart style: the repo's palette ──────────────────────────────────────────
# Verdicts: a validated blue/red pair plus a neutral gray for "missed" (deliberately
# colourless — it means nothing was extracted). Models: the same slot order every
# other notebook in this repo uses, so a model's colour never changes.
VERDICT_COLOUR = {"Correct": "#2a78d6", "Hallucinated": "#e34948", "Missed": "#898781"}
MODEL_COLOUR   = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#d8d8d4", "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.grid": False, "legend.frameon": False,
})
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)

samples = sorted(int(m.group(1)) for p in RDA_DIR.glob("RDA_DMP_sample*_manual_annotation.json")
                 if (m := re.search(r"sample(\d+)", p.name)))
print("samples with a reference:", samples)
for n in samples:
    have = [m for m in MODELS if (RDA_DIR / OUTPUT.format(n=n, model=m)).exists()]
    print(f"  sample{n}: outputs from {', '.join(have) or 'no model'}")
'''),

md("s1", '''
## 1. Flatten the JSON into fields

A nested document becomes a list of `path = value` pairs, e.g.
`dmp.dataset[2].title = "Otolith Microchemistry from juvenile sockeye"`.

- Where the schema allows *either one object or a list* (`contact_id`, `contributor_id`,
  `creator_id`, `metadata_standard_id`), a single object is treated as a one-item list, so
  `contact_id.identifier` and `contact_id[0].identifier` are the same field.
- A reference field only counts if it has a value. An empty model field still counts as
  "output", because the model chose to write it.
'''),

code("flatten", r'''
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


def load(path):
    return canonical(json.loads(Path(path).read_text(encoding="utf-8")))


ref = load(RDA_DIR / REFERENCE.format(n=samples[0]))
ref_fields = {p: v for p, v in flatten(ref) if not is_empty(v)}
print(f"sample{samples[0]} reference: {len(ref_fields)} fields with a value")
for p, v in list(ref_fields.items())[:6]:
    print(f"  {p:48} {json.dumps(v, ensure_ascii=False)[:60]}")
print("  ...")
'''),

md("s2", '''
## 2. Line up list items

A model may list the datasets in a different order from the reference, or find only some
of them. Each list item in the model output is paired with a reference item — by matching
**title / name / identifier** when one exists, otherwise by **position** — and the model's
paths are rewritten to use the reference's indices. So `dataset[1]` in the model can be
scored against `dataset[2]` in the reference when that is the dataset it describes.
'''),

code("align", r'''
KEY_FIELDS = ("title", "name", "identifier", "license_ref")


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
    """model index -> reference index. Exact key matches first, then position."""
    ref_keys = {item_key(r): j for j, r in enumerate(ref_list)}
    ref_keys.pop(None, None)
    mapping, taken = {}, set()
    for i, m in enumerate(model_list):                       # pass 1: same title / name
        j = ref_keys.get(item_key(m))
        if j is not None and j not in taken:
            mapping[i], taken = j, taken | {j}
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


# How each model's datasets were paired, for the first sample
ds_ref = ref.get("dmp", {}).get("dataset", [])
for m in MODELS:
    p = RDA_DIR / OUTPUT.format(n=samples[0], model=m)
    if not p.exists():
        continue
    ds_model = load(p).get("dmp", {}).get("dataset", [])
    print(f"{m}:")
    for i, j in pair_items(ds_model, ds_ref).items():
        mt = ds_model[i].get("title") if isinstance(ds_model[i], dict) else ds_model[i]
        rt = ds_ref[j].get("title") if j < len(ds_ref) else "(no reference item)"
        print(f"   model dataset[{i}] {str(mt)[:36]!r:40} -> reference dataset[{j}] {str(rt)[:36]!r}")
'''),

md("s3", '''
## 3. Judge each field

`verdict(path, model_value, reference_value)` returns **Correct** or **Hallucinated** and a
one-line reason, using the rule for that kind of field (the table at the top). The examples
underneath show the rules on real cases from sample 14.
'''),

code("judge", r'''
def field_kind(path):
    last = re.sub(r"\[\d+\]$", "", path.split(".")[-1])
    if last in ID_FIELDS:   return "identifier"
    if last in DATE_FIELDS: return "date"
    if last in ENUM_FIELDS: return "controlled"
    return "text"


LICENCE = [(re.compile(r"creativecommons\.org/licenses/([a-z\-]+)/(\d\.\d)"), r"cc-\1-\2"),
           (re.compile(r"^cc[\s\-]+([a-z][a-z\-]*)[\s\-]+(\d\.\d)$"), r"cc-\1-\2")]


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


def verdict(path, mv, rv):
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


examples = [
    ("dmp.dataset[0].title", "Hakai JSP Time Series", "Hakai Institute Juvenile Salmon Program Time Series"),
    ("dmp.title", "Hakai Institute Juvenile Salmon Program Time Series Data Management Plan",
                  "Hakai Institute Juvenile Salmon Program Time Series"),
    ("dmp.dataset[0].distribution[0].license[0].license_ref", "https://creativecommons.org/licenses/by/4.0/", "CC BY 4.0"),
    ("dmp.project[0].start", "2015-05-12T00:00:00Z", "2015-05-12"),
    ("dmp.contact.mbox", "brett.johnson@hakai.org", None),
    ("dmp.contact.mbox", "N/A", None),
    ("dmp.contact.name", "N/A", "Brett Johnson"),
    ("dmp.dataset[0].dataset_id.identifier", "https://doi.org/10.48321/D1CW23", "https://doi.org/10.21966/1.566666"),
]
print(f"{'verdict':13} {'model value':46} {'reference value':38} reason")
for path, mv, rv in examples:
    v, why = verdict(path, mv, rv)
    print(f"{v:13} {str(mv)[:42]!r:46} {str(rv)[:34]!r:38} {why}")
'''),

md("s4", '''
## 4. Score every sample and model

One row per judged field. Missed reference fields are added as rows with no model value.
'''),

code("score", r'''
rows = []
for n in samples:
    ref = load(RDA_DIR / REFERENCE.format(n=n))
    ref_values = dict(flatten(ref))
    ref_fields = {p for p, v in ref_values.items() if not is_empty(v)}
    for m in MODELS:
        p = RDA_DIR / OUTPUT.format(n=n, model=m)
        if not p.exists():
            continue
        covered = set()
        for path, mpath, mv in flatten_aligned(load(p), ref):
            rv = ref_values.get(path)
            v, why = verdict(path, mv, rv)
            if v == "Correct" and path in ref_fields:
                covered.add(path)
            rows.append({"sample": n, "model": m, "field": path, "model field": mpath, "model value": mv,
                         "reference value": rv, "verdict": v, "reason": why})
        for path in sorted(ref_fields - covered):
            rows.append({"sample": n, "model": m, "field": path, "model field": None, "model value": None,
                         "reference value": ref_values[path], "verdict": "Missed", "reason": ""})

details = pd.DataFrame(rows)
print(f"{len(details)} judged rows across {len(samples)} sample(s) and {details['model'].nunique()} models\n")
details.groupby(["model", "verdict"]).size().unstack(fill_value=0)[["Correct", "Hallucinated", "Missed"]].loc[
    [m for m in MODELS if m in set(details["model"])]]
'''),

md("s5", '''
## 5. Metrics

One row per model, over all samples. `fields in reference` counts reference fields with a
value; `fields output` counts every field the model wrote.
'''),

code("metrics", r'''
def metrics(df, n_ref):
    c = int((df["verdict"] == "Correct").sum())
    h = int((df["verdict"] == "Hallucinated").sum())
    miss = int((df["verdict"] == "Missed").sum())
    out, found, total = c + h, n_ref - miss, c + h + miss
    precision = c / out if out else 0.0
    recall = found / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"fields in reference": n_ref, "fields output": out, "correct": c, "hallucinated": h,
            "missed": miss, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "correct %": c / total, "hallucinated %": h / total, "missed %": miss / total}


n_ref = {n: len({p for p, v in flatten(load(RDA_DIR / REFERENCE.format(n=n))) if not is_empty(v)}) for n in samples}
summary = pd.DataFrame([{"model": m, **metrics(details[details["model"] == m], sum(n_ref.values()))}
                        for m in MODELS if m in set(details["model"])]).set_index("model")

def show(df):
    """The metrics table with the shares as percentages."""
    out = df.copy()
    for c in ("correct %", "hallucinated %", "missed %"):
        out[c] = out[c].map("{:.1%}".format)
    display(out)


show(summary)

if len(samples) > 1:
    per_sample = pd.DataFrame([{"sample": n, "model": m, **metrics(details[(details["model"] == m) & (details["sample"] == n)], n_ref[n])}
                               for n in samples for m in MODELS if ((details["model"] == m) & (details["sample"] == n)).any()]
                              ).set_index(["sample", "model"])
    print("\nPer sample:")
    show(per_sample)

# The arithmetic behind the three numbers, with this run's values for the first model
m = summary.index[0]
c, out, ref_n, miss = (int(summary.loc[m, k]) for k in ("correct", "fields output", "fields in reference", "missed"))
print(f"\nWorked example, {m}:")
print(f"  precision = correct / fields output              = {c} / {out} = {summary.loc[m, 'precision']}")
print(f"  recall    = (reference - missed) / reference     = ({ref_n} - {miss}) / {ref_n} = {summary.loc[m, 'recall']}")
print(f"  F1        = 2 * precision * recall / (precision + recall) = {summary.loc[m, 'f1']}")
'''),

md("s6", '''
## 6. Detailed comparison, per model

Every field the model wrote — model value, reference value, verdict, reason — with the
Hallucinated rows first so the problems are at the top. (Missed fields are counted above and
listed in the saved workbook, section 9.)
'''),

code("tables", r'''
ORDER = {"Hallucinated": 0, "Correct": 1}
for m in MODELS:
    d = details[(details["model"] == m) & (details["verdict"] != "Missed")].copy()
    if d.empty:
        continue
    d = d.sort_values(["sample", "verdict", "model field"], key=lambda col: col.map(ORDER) if col.name == "verdict" else col)
    counts = details[details["model"] == m]["verdict"].value_counts()
    print(f"\n{'=' * 100}\n{m}: {counts.get('Correct', 0)} correct, {counts.get('Hallucinated', 0)} hallucinated, "
          f"{counts.get('Missed', 0)} missed\n{'=' * 100}")
    display(d[["sample", "model field", "model value", "reference value", "verdict", "reason"]]
            .rename(columns={"model field": "field"}).reset_index(drop=True))
'''),

md("s7", '''
## 7. Where the misses are

Missed fields grouped by the part of the schema they belong to, next to how many reference
fields that part has — this says *what kind* of information each model fails to extract.
'''),

code("missed", r'''
def section(path):
    parts = path.split(".")
    return re.sub(r"\[\d+\]", "", parts[1]) if parts[0] == "dmp" and len(parts) > 1 else parts[0]


ref_counts = pd.Series([section(p) for n in samples for p, v in flatten(load(RDA_DIR / REFERENCE.format(n=n)))
                        if not is_empty(v)]).value_counts()
missed = details[details["verdict"] == "Missed"].copy()
missed["section"] = missed["field"].map(section)
by_section = (missed.groupby(["section", "model"]).size().unstack(fill_value=0)
              .reindex(ref_counts.index, fill_value=0)[[m for m in MODELS if m in set(details["model"])]])
by_section.insert(0, "reference fields", ref_counts)
by_section.columns.name = "missed by"
by_section
'''),

md("s8", '''
## 8. Charts

Left: each model's fields split into correct, hallucinated and missed, with the counts in
the label so nothing depends on a segment being wide enough to read. Right: precision,
recall and F1 per model.
'''),

code("charts", r'''
models = list(summary.index)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

# ── Shares, one bar per model ─────────────────────────────────────────────────
labels = [f"{m}\n{int(summary.loc[m, 'correct'])} correct · {int(summary.loc[m, 'hallucinated'])} hallucinated · "
          f"{int(summary.loc[m, 'missed'])} missed" for m in models]
left = [0.0] * len(models)
for v in ("Correct", "Hallucinated", "Missed"):
    vals = [summary.loc[m, f"{v.lower()} %"] for m in models]
    bars = ax1.barh(labels, vals, left=left, height=0.52, color=VERDICT_COLOUR[v],
                    edgecolor=SURFACE, linewidth=2, label=v)
    for bar, val in zip(bars, vals):
        if val >= 0.05:                                       # percentage inside the segment when it fits
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                     ha="center", va="center", fontsize=8.5, color="white" if v != "Missed" else INK)
    left = [l + x for l, x in zip(left, vals)]
ax1.set_xlim(0, 1)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax1.invert_yaxis()
ax1.tick_params(axis="y", labelsize=9)
ax1.set_title("Fields: correct, hallucinated, missed")
ax1.grid(axis="x", color=GRID, linewidth=0.9)
ax1.set_axisbelow(True)
ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
for side in ("top", "right", "left"):
    ax1.spines[side].set_visible(False)

# ── Precision / recall / F1 per model ────────────────────────────────────────
shown = ["precision", "recall", "f1"]
w = 0.8 / len(models)
for k, m in enumerate(models):
    xs = [i + (k - (len(models) - 1) / 2) * w for i in range(len(shown))]
    bars = ax2.bar(xs, [summary.loc[m, s] for s in shown], width=w * 0.92,
                   color=MODEL_COLOUR[m], edgecolor=SURFACE, linewidth=2, label=m)
    ax2.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=MUTED)
ax2.set_xticks(range(len(shown)))
ax2.set_xticklabels(["Precision", "Recall", "F1"])
ax2.set_ylim(0, 1.08)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_title("Precision, recall and F1")
ax2.grid(axis="y", color=GRID, linewidth=0.9)
ax2.set_axisbelow(True)
ax2.legend(ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.14))
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
fig.savefig(CHART, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.show()
'''),

md("s9", '''
## 9. Save

An Excel workbook with a `summary` sheet, one sheet per model listing every verdict
(including the Missed fields), and `details` with all rows together.
'''),

code("save", r'''
with pd.ExcelWriter(RESULTS) as xw:
    summary.to_excel(xw, sheet_name="summary")
    for m in MODELS:
        d = details[details["model"] == m]
        if not d.empty:
            d.drop(columns="model").to_excel(xw, sheet_name=m[:31], index=False)
    details.to_excel(xw, sheet_name="details", index=False)
print(f"saved -> {RESULTS}")
print(f"saved -> {CHART}")
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
