"""Build notebooks/rda-dmp-json.ipynb.

One notebook for the whole loop: the pdfplumber text of sample 14 plus the complete
maDMP 1.2 schema go to llama3.1:8b, gemma4:e4b and llama3.3:70b; the results are
scored against the hand-made reference; every saved prompt version is compared.

Each prompt version is a named run under data/output/rda/runs/<RUN_NAME>/. Running
the notebook again with the same name re-uses the saved results (nothing is
re-generated); changing the prompt without changing the name stops with a message.

    python scripts/build/build_rda_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/rda-dmp-json.ipynb")


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
# PDF → RDA DMP JSON: run a prompt version, score it, compare versions

The text of one DMP PDF plus the complete **maDMP 1.2** schema go to three models.
What comes back is scored against the form a person filled in by hand from the same
PDF. Every prompt version is saved under its own name, so you can see what a change
to the prompt did.

**How to try a new prompt**

1. Edit the prompt in Step 3.
2. Give the version a new name: `RUN_NAME` in the settings cell (`v1`, `v2`, ...).
3. Run all cells. Step 7 shows this version next to every earlier one.

Running again with an existing name does not call the models again — the saved results
are loaded and re-scored. If the prompt was changed but the name was not, Step 3 stops
and asks for a new name, so no version is ever overwritten.

| Step | What happens |
|---|---|
| 1 | Read sample 14 with pdfplumber |
| 2 | Load the schema — unchanged |
| 3 | The prompt — the only thing to edit between versions |
| 4 | Run the three models, save each result as it comes |
| 5 | Score this version: correct, hallucinated, missed; precision, recall, F1 |
| 6 | Every field, model by model |
| 7 | Compare every saved version — scores, what changed field by field, and the prompt change |

**The marks.** Every field a model wrote is **Correct** (matches the person's value, or both
blank) or **Hallucinated** (wrong, made up, or blank where the person had a value). A field the
person filled in that the model never got right is **Missed**. Precision is the share of what
the model wrote that was right; recall is the share of the document's fields it found; F1
balances the two. Identifiers must match exactly (ignoring `https://` and case), dates by
day, a licence by name or URL, and free text counts when at least three-quarters of the
model's words appear in the person's version.
'''),

code("setup", r'''
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import requests

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.models.ollama import OllamaModel
from dmpbridge.evaluation.evaluate import tokenize, containment, CONTAINMENT_THRESHOLD

# ── This version ─────────────────────────────────────────────────────────────
RUN_NAME = "v2"                        # change this every time you change the prompt

# ── Inputs ───────────────────────────────────────────────────────────────────
SAMPLE    = 14
PDF       = Path("data/input/pdfs") / f"sample{SAMPLE}.pdf"
SCHEMA    = Path("data/output/rda/maDMP-schema-1.2.json")
REFERENCE = Path("data/output/rda") / f"RDA_DMP_sample{SAMPLE}_manual_annotation.json"

# ── Models ───────────────────────────────────────────────────────────────────
HOST       = "http://localhost:11434"
NUM_CTX    = 32768
MAX_TOKENS = 8000                      # cap on the answer, in case a model loops and never stops
MODEL_1 = "llama3.1:8b"
MODEL_2 = "gemma4:e4b"
MODEL_3 = "llama3.3:70b"
MODELS  = [MODEL_1, MODEL_2, MODEL_3]

# ── Outputs ──────────────────────────────────────────────────────────────────
RUNS_DIR = Path("data/output/rda/runs")
RUN_DIR  = RUNS_DIR / RUN_NAME
tag = lambda model: model.replace(":", "-")                       # model name as used in file names

# ── Judging rules ────────────────────────────────────────────────────────────
EMPTY       = {"", "null", "none", "n/a", "na"}
THRESHOLD   = CONTAINMENT_THRESHOLD                                # 0.75
ONE_OR_MANY = {"contact_id", "contributor_id", "creator_id", "metadata_standard_id"}
ID_FIELDS   = {"identifier", "mbox", "url", "access_url", "download_url", "scheme_uri", "license_ref"}
DATE_FIELDS = {"created", "modified", "issued", "start", "end", "start_date", "available_until"}
ENUM_FIELDS = {"type", "data_access", "personal_data", "sensitive_data", "ethical_issues_exist",
               "language", "relation_type", "resource_type", "funding_status", "role",
               "certified_with", "geo_location", "pid_system", "currency_code", "is_reused"}

# ── Chart style: the repo's palette ──────────────────────────────────────────
VERDICT_COLOUR = {"Correct": "#2a78d6", "Hallucinated": "#e34948", "Missed": "#898781"}
MODEL_COLOUR   = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#d8d8d4", "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.grid": False, "legend.frameon": False,
})
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)

saved_runs = sorted(p.name for p in RUNS_DIR.glob("*") if p.is_dir())
print(f"this version: {RUN_NAME!r}  ({'already saved - results will be loaded' if RUN_DIR.exists() else 'new'})")
print("saved versions:", ", ".join(saved_runs) or "none yet")
'''),

md("s1", '''
## Step 1 — Read the PDF
'''),

code("read", '''
dmp_text = get_extractor("pdfplumber").extract(PDF)[0]["text"]
print(f"sample{SAMPLE}: {len(dmp_text):,} characters")
print(dmp_text[:300])
'''),

md("s2", '''
## Step 2 — Load the schema

The schema is used exactly as published — nothing removed, nothing changed. The
`$ref` pointers are resolved so the model sees every definition in place.
'''),

code("schema", '''
schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
defs = schema["$defs"]


def resolve_refs(node):
    """Replace every $ref with the definition it points at. Content unchanged."""
    if isinstance(node, dict):
        if "$ref" in node:
            return resolve_refs(defs[node["$ref"].split("/")[-1]])
        return {k: resolve_refs(v) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [resolve_refs(v) for v in node]
    return node


schema_full = resolve_refs(schema)
schema_text = json.dumps(schema_full, separators=(",", ":"))
print(f"{len(schema_text):,} characters of schema, {len(defs)} definitions resolved")
'''),

md("s3", '''
## Step 3 — The prompt

**This is the cell to edit.** The schema is given to the model twice: as text in the
prompt, and as Ollama's `format`, which constrains the JSON it generates to that
structure.

The exact prompt is saved as `prompt.txt` in the version's folder. If it differs from
the one already saved under this `RUN_NAME`, the cell stops and asks for a new name.
'''),

code("prompt", r'''
SYSTEM = """You convert Data Management Plans into RDA maDMP JSON.

Strict rules:
1. Use only the field names defined in the schema. Never add a key that is not in the schema.
2. Put every field exactly where the schema places it. The whole document is one top-level "dmp" object.
3. Where the schema lists allowed values, use one of them, spelled exactly as in the schema .
4. Take every value from the Data Management Plan text. Never copy example values from the schema and never hallucinate.
5. Output only the JSON object. No explanation, no markdown.
6. Include every dataset, contributor, funder, host and distribution the text mentions. Leave out any field the text gives no information for."""


def make_prompt(dmp_text):
    return f"""Here is the RDA maDMP JSON schema (version 1.2). Follow it strictly:

{schema_text}

Here is the text of a Data Management Plan:

{dmp_text}

Generate one JSON object that strictly follows the schema above, filling in
whatever information the Data Management Plan text contains. Output only the JSON."""


# The prompt on record for this version (schema and document text shown as placeholders).
prompt_record = ("SYSTEM:\n" + SYSTEM + "\n\nPROMPT (the schema and the document text go in the braces):\n"
                 + make_prompt("{dmp_text}").replace(schema_text, "{schema_text}"))

saved_prompt = RUN_DIR / "prompt.txt"
if saved_prompt.exists() and saved_prompt.read_text(encoding="utf-8") != prompt_record:
    raise SystemExit(f"The prompt is not the one saved for version {RUN_NAME!r}.\n"
                     f"Give this prompt a new RUN_NAME in the settings cell (saved versions: {', '.join(saved_runs)}),\n"
                     f"or delete {RUN_DIR} if you really want to redo {RUN_NAME!r}.")
RUN_DIR.mkdir(parents=True, exist_ok=True)
saved_prompt.write_text(prompt_record, encoding="utf-8")
print(f"prompt saved -> {saved_prompt}")
print(f"{len(make_prompt(dmp_text)):,} characters go to the model: the system text, the schema and the document")
'''),

md("s4", '''
## Step 4 — Run the three models

One model at a time: it is loaded, checked to be **100% on the GPU** (a model spilling
onto the CPU turns seconds into hours and is skipped), then called with the prompt. Each
result is saved as soon as it arrives, with the time taken and the tokens in and out.
A model whose result is already saved under this `RUN_NAME` is not called again.
A model that never finishes a valid JSON (it looped until the token cap) is recorded as
failed for this version, and what it wrote is kept in a `.raw.txt` file next to the results.

The 70B needs about 53 GB of VRAM at this context size — all three 24 GB cards. With
fewer cards available it cannot sit fully on the GPU and is skipped.
'''),

code("run", r'''
RUN_INFO = RUN_DIR / "run.json"
timings = {k.split(" sample")[0]: v for k, v in                       # keyed by model name
           (json.loads(RUN_INFO.read_text(encoding="utf-8")).get("timings", {}) if RUN_INFO.exists() else {}).items()}
results, not_run = {}, {}            # not_run: model -> why there is no result


def save_run_info():
    RUN_INFO.write_text(json.dumps({
        "run": RUN_NAME, "date": datetime.now().isoformat(timespec="minutes"), "sample": SAMPLE,
        "models": MODELS, "schema": SCHEMA.name, "num_ctx": NUM_CTX, "timings": timings, "not_run": not_run,
    }, indent=2), encoding="utf-8")


def run(model):
    """One model on the prompt. Returns the JSON, or None if the model had to be skipped."""
    out = RUN_DIR / f"sample{SAMPLE}.rda.{tag(model)}.json"
    if out.exists():
        print(f"{model}: already saved for version {RUN_NAME!r}, loaded from {out}")
        t = timings.get(model, {})
        if t:
            print(f"(that run took {t['seconds']} s, {t['tokens_sent']:,} tokens sent, {t['tokens_generated']:,} generated)")
        return json.loads(out.read_text(encoding="utf-8"))

    for other in MODELS:
        if other != model:
            subprocess.run(["ollama", "stop", other], check=False)   # one model in VRAM at a time

    # Load the model, then make sure all of it is on the GPU before the real call.
    t0 = time.perf_counter()
    requests.post(f"{HOST}/api/generate", timeout=1800,
                  json={"model": model, "keep_alive": -1, "options": {"num_ctx": NUM_CTX}})
    load = time.perf_counter() - t0
    loaded = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    placement = next((l for l in loaded.splitlines() if l.startswith(model)), "")
    if "100% GPU" not in placement:
        not_run[model] = f"not fully on the GPU (ollama ps says: {' '.join(placement.split()) or 'not loaded'})"
        print(f"{model}: SKIPPED - {not_run[model]}")
        subprocess.run(["ollama", "stop", model], check=False)
        return None

    llm = OllamaModel(model=model, host=HOST, num_ctx=NUM_CTX, num_predict=MAX_TOKENS)
    t0 = time.perf_counter()
    raw = llm.complete(SYSTEM, make_prompt(dmp_text), schema=schema_full)
    elapsed = time.perf_counter() - t0

    s = llm.last_call
    timings[model] = {"seconds": round(elapsed), "model_load_seconds": round(load),
                      "tokens_sent": s["prompt_eval_count"], "tokens_generated": s["eval_count"]}
    print(f"{model}: {elapsed:.0f} s   (loading the model took {load:.0f} s)")
    print(f"tokens sent to the model: {s['prompt_eval_count']:,}   tokens generated: {s['eval_count']:,}")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        # The model did not finish a valid JSON - usually it looped and hit the token cap.
        cut = (f"it stopped at the {MAX_TOKENS:,}-token cap, so it never finished"
               if s["done_reason"] == "length" else f"{e.msg} at character {e.pos:,}")
        not_run[model] = f"no valid JSON - {cut}"
        raw_file = RUN_DIR / f"sample{SAMPLE}.rda.{tag(model)}.raw.txt"
        raw_file.write_text(raw, encoding="utf-8")
        print(f"{model}: FAILED - {not_run[model]}")
        print(f"what it wrote is kept in {raw_file}")
        return None
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")
    return result


def run_and_show(model):
    results[model] = run(model)
    save_run_info()
    if results[model] is not None:
        print()
        print(json.dumps(results[model], indent=2, ensure_ascii=False))


run_and_show(MODEL_1)
'''),

code("gemma", '''
run_and_show(MODEL_2)
'''),

code("llama33", '''
run_and_show(MODEL_3)
'''),

md("s5", '''
## Step 5 — Score this version

Each model's JSON is compared field by field with the reference. The form is nested, so
every entry becomes one line (field name, value); a model's datasets and contributors are
paired with the person's by title or name, otherwise by position. Then each value gets its
mark by the rules at the top.
'''),

code("judge", r'''
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


def norm_text(v):
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def item_key(item):
    """The value that identifies a list item, if it has one."""
    if isinstance(item, dict):
        for k in ("title", "name", "identifier", "license_ref"):
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
            yield from flatten_aligned(v, ref_list[j] if j < len(ref_list) else None, f"{path}[{j}]", f"{mpath}[{i}]")
    else:
        yield path, mpath, model


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


def score(model_json, ref):
    """Every judged field of one model's output as rows."""
    ref_values = dict(flatten(ref))
    ref_fields = {p for p, v in ref_values.items() if not is_empty(v)}
    rows, covered = [], set()
    for path, mpath, mv in flatten_aligned(canonical(model_json), ref):
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


def metrics(df, n_ref):
    c = int((df["verdict"] == "Correct").sum())
    h = int((df["verdict"] == "Hallucinated").sum())
    miss = int((df["verdict"] == "Missed").sum())
    out, found = c + h, n_ref - miss
    precision = c / out if out else 0.0
    recall = found / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"fields in reference": n_ref, "fields output": out, "correct": c, "hallucinated": h, "missed": miss,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


ref = canonical(json.loads(REFERENCE.read_text(encoding="utf-8")))
n_ref = len({p for p, v in flatten(ref) if not is_empty(v)})

rows = []
for model, result in results.items():
    if result is not None:
        rows += [{"model": model, **r} for r in score(result, ref)]
details = pd.DataFrame(rows)
for m, why in not_run.items():
    print(f"not scored: {m} - {why}")
if details.empty:
    raise SystemExit(f"No model produced a result for version {RUN_NAME!r}, so there is nothing to score.")

scored = [m for m in MODELS if m in set(details["model"])]
summary = pd.DataFrame([{"model": m, **metrics(details[details["model"] == m], n_ref)} for m in scored]).set_index("model")
print(f"version {RUN_NAME!r}, sample{SAMPLE}: the person filled in {n_ref} fields\n")
display(summary)

m = summary.index[0]
c, out, miss = (int(summary.loc[m, k]) for k in ("correct", "fields output", "missed"))
print(f"\nWorked example, {m}:")
print(f"  precision = correct / fields output          = {c} / {out} = {summary.loc[m, 'precision']}")
print(f"  recall    = (reference - missed) / reference = ({n_ref} - {miss}) / {n_ref} = {summary.loc[m, 'recall']}")
print(f"  F1        = 2 * precision * recall / (precision + recall) = {summary.loc[m, 'f1']}")

# Saved with the version, so Step 7 can compare it with the others later
(RUN_DIR / "scores.json").write_text(json.dumps({
    "run": RUN_NAME, "date": datetime.now().isoformat(timespec="minutes"), "sample": SAMPLE,
    "fields in reference": n_ref, "scores": summary.to_dict(orient="index")}, indent=2), encoding="utf-8")
details.to_csv(RUN_DIR / "details.csv", index=False)
with pd.ExcelWriter(RUN_DIR / "evaluation_results.xlsx") as xw:
    summary.to_excel(xw, sheet_name="summary")
    for m in scored:
        details[details["model"] == m].drop(columns="model").to_excel(xw, sheet_name=tag(m)[:31], index=False)
print(f"\nsaved -> {RUN_DIR / 'scores.json'}, details.csv and evaluation_results.xlsx")
'''),

code("chart", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

# ── Shares, one bar per model ─────────────────────────────────────────────────
labels = [f"{m}\n{int(summary.loc[m, 'correct'])} correct · {int(summary.loc[m, 'hallucinated'])} hallucinated · "
          f"{int(summary.loc[m, 'missed'])} missed" for m in scored]
totals = {m: int(summary.loc[m, ["correct", "hallucinated", "missed"]].sum()) for m in scored}
left = [0.0] * len(scored)
for v in ("Correct", "Hallucinated", "Missed"):
    vals = [summary.loc[m, v.lower()] / totals[m] for m in scored]
    bars = ax1.barh(labels, vals, left=left, height=0.52, color=VERDICT_COLOUR[v], edgecolor=SURFACE, linewidth=2, label=v)
    for bar, val in zip(bars, vals):
        if val >= 0.05:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                     ha="center", va="center", fontsize=8.5, color="white" if v != "Missed" else INK)
    left = [l + x for l, x in zip(left, vals)]
ax1.set_xlim(0, 1)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax1.invert_yaxis()
ax1.tick_params(axis="y", labelsize=9)
ax1.set_title(f"Version {RUN_NAME}: fields correct, hallucinated, missed")
ax1.grid(axis="x", color=GRID, linewidth=0.9)
ax1.set_axisbelow(True)
ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
for side in ("top", "right", "left"):
    ax1.spines[side].set_visible(False)

# ── Precision / recall / F1 per model ────────────────────────────────────────
shown = ["precision", "recall", "f1"]
w = 0.8 / len(scored)
for k, m in enumerate(scored):
    xs = [i + (k - (len(scored) - 1) / 2) * w for i in range(len(shown))]
    bars = ax2.bar(xs, [summary.loc[m, s] for s in shown], width=w * 0.92, color=MODEL_COLOUR[m],
                   edgecolor=SURFACE, linewidth=2, label=m)
    ax2.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=MUTED)
ax2.set_xticks(range(len(shown)))
ax2.set_xticklabels(["Precision", "Recall", "F1"])
ax2.set_ylim(0, 1.08)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_title("Precision, recall and F1")
ax2.grid(axis="y", color=GRID, linewidth=0.9)
ax2.set_axisbelow(True)
ax2.legend(ncol=len(scored), loc="upper center", bbox_to_anchor=(0.5, -0.14))
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

plt.tight_layout()
fig.savefig(RUN_DIR / "evaluation_charts.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.show()
'''),

md("s6", '''
## Step 6 — Every field, model by model

What each model wrote next to what the person wrote, with the mark and the reason.
Hallucinated rows come first so the problems are at the top. Missed fields are listed
in the Excel file saved with the version.
'''),

code("tables", r'''
ORDER = {"Hallucinated": 0, "Correct": 1}
for m in scored:
    d = details[(details["model"] == m) & (details["verdict"] != "Missed")].copy()
    d = d.sort_values(["verdict", "model field"], key=lambda col: col.map(ORDER) if col.name == "verdict" else col)
    counts = details[details["model"] == m]["verdict"].value_counts()
    print(f"\n{'=' * 100}\n{m}: {counts.get('Correct', 0)} correct, {counts.get('Hallucinated', 0)} hallucinated, "
          f"{counts.get('Missed', 0)} missed\n{'=' * 100}")
    display(d[["model field", "model value", "reference value", "verdict", "reason"]]
            .rename(columns={"model field": "field"}).reset_index(drop=True))
'''),

md("s7", '''
## Step 7 — Compare every saved version

Every version under `data/output/rda/runs/` that has been scored, oldest first. The
table and chart show the scores per version and model. Below them: what changed field by
field between this version and the one before it, and the change made to the prompt.
'''),

code("compare", r'''
history = []
for d in RUNS_DIR.iterdir():
    if d.is_dir() and (d / "scores.json").exists():
        info = json.loads((d / "scores.json").read_text(encoding="utf-8"))
        for m, s in info["scores"].items():
            history.append({"version": d.name, "date": info["date"], "model": m, **s})
history = pd.DataFrame(history).sort_values(["date", "version"])
versions = list(dict.fromkeys(history["version"]))

print("versions, oldest first:", ", ".join(versions))
table = history.set_index(["version", "model"])[["correct", "hallucinated", "missed", "precision", "recall", "f1"]]
display(table.loc[versions])

print("\nF1 per version and model:")
display(history.pivot(index="version", columns="model", values="f1").loc[versions].reindex(columns=[m for m in MODELS if m in set(history["model"])]))

# ── Chart: F1 and recall across versions, one line per model ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, metric in zip(axes, ("f1", "recall")):
    for m in MODELS:
        h = history[history["model"] == m].set_index("version").reindex(versions)
        if h[metric].notna().any():
            ax.plot(range(len(versions)), h[metric], marker="o", markersize=7, linewidth=2, color=MODEL_COLOUR[m], label=m)
            for x, y in enumerate(h[metric]):
                if pd.notna(y):
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions)
    ax.set_ylim(0, 1.08)
    ax.set_title({"f1": "F1 by version", "recall": "Recall by version"}[metric])
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(loc="upper left")
plt.tight_layout()
fig.savefig(RUNS_DIR / "versions.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.show()
'''),

code("diff", r'''
import difflib

i = versions.index(RUN_NAME)
previous = versions[i - 1] if i > 0 else None
if previous is None:
    print(f"{RUN_NAME!r} is the only scored version - nothing to compare with yet.")
else:
    print(f"What changed from {previous!r} to {RUN_NAME!r}\n")

    def status_per_field(df):
        """One mark per reference field: Correct beats Hallucinated beats Missed."""
        rank = {"Correct": 0, "Hallucinated": 1, "Missed": 2}
        return df.assign(r=df["verdict"].map(rank)).sort_values("r").drop_duplicates("field").set_index("field")["verdict"]

    before_all = pd.read_csv(RUNS_DIR / previous / "details.csv")
    for m in scored:
        if m not in set(before_all["model"]):
            print(f"{m}: not in {previous!r}, so no comparison\n")
            continue
        before = status_per_field(before_all[before_all["model"] == m])
        after = status_per_field(details[details["model"] == m])
        both = pd.concat([before.rename("before"), after.rename("after")], axis=1).fillna("Missed")
        changed = both[both["before"] != both["after"]]
        fixed = changed[changed["after"] == "Correct"]
        broken = changed[changed["before"] == "Correct"]
        other = changed.drop(index=fixed.index).drop(index=broken.index)
        s_prev = history[(history["version"] == previous) & (history["model"] == m)].iloc[0]
        s_now = summary.loc[m]
        print(f"{m}: F1 {s_prev['f1']:.3f} -> {s_now['f1']:.3f}   "
              f"correct {int(s_prev['correct'])} -> {int(s_now['correct'])}, "
              f"hallucinated {int(s_prev['hallucinated'])} -> {int(s_now['hallucinated'])}, "
              f"missed {int(s_prev['missed'])} -> {int(s_now['missed'])}")
        print(f"   now right, was not:   {len(fixed)}" + (f"   {', '.join(fixed.index)}" if len(fixed) else ""))
        print(f"   was right, now not:   {len(broken)}" + (f"   {', '.join(broken.index)}" if len(broken) else ""))
        if len(other):
            print(f"   missed <-> hallucinated: {len(other)}   {', '.join(other.index)}")
        print()

    print("Prompt change:\n")
    diff = list(difflib.unified_diff(
        (RUNS_DIR / previous / "prompt.txt").read_text(encoding="utf-8").splitlines(),
        prompt_record.splitlines(), fromfile=previous, tofile=RUN_NAME, lineterm="", n=1))
    print("\n".join(diff) if diff else "   none - the same prompt (settings such as the model or context may differ)")
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
