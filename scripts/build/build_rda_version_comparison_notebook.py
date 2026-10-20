"""Build notebooks/rda-version-comparison.ipynb.

Several saved RDA prompt versions side by side, for one sample: per model and summed
over the models, how many annotated fields each version got Correct, Mismatch or
Missed; then the same split by part of the form. Reads the runs' saved JSON only;
no model is called. Scoring is dmpbridge.rda.evaluate, the same code the pipeline
and the other RDA notebooks use.

    python scripts/build/build_rda_version_comparison_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/rda-version-comparison.ipynb")


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
# RDA prompt versions compared

One sample, several saved prompt versions, three models. For every field the person
annotated, each version's output is **Correct** (same value), a **Mismatch** (a
different value) or **Missed** (nothing written there) - the same labels as the field
matrix. Lower mismatch and missed are better; a mismatch is the costlier error, since
it is a wrong value that looks right.

**To compare other versions or another sample:** change `RUNS` and `SAMPLE` in the next cell.
'''),

code("setup", r'''
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.rda import evaluate as E

SAMPLE = 3
# (label shown in the figure, folder under notebooks/rda-experiments/runs/)
RUNS = [("v9",                "v9-sample3"),
        ("v10",               "v10-sample3"),
        ("v11",               "v11-sample3"),
        ("v12",               "v12-sample3")]
MODELS = ["llama3.1-8b", "gemma4-e4b", "llama3.3-70b"]
NAME   = {"llama3.1-8b": "Llama 3.1 8B", "gemma4-e4b": "Gemma 4 e4b", "llama3.3-70b": "Llama 3.3 70B"}
OUTCOMES = ["Correct", "Mismatch", "Missed"]
COLOUR = {"Correct": "#238b45", "Mismatch": "#e34948", "Missed": "#898781"}
OUT_DIR = Path("notebooks/rda-experiments")

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                     "axes.edgecolor": "#d8d8d4", "legend.frameon": False})
pd.set_option("display.width", 200)
'''),

md("s1", '''
## 1. Score every version
'''),

code("score", r'''
ref = E.reference_path(SAMPLE)
rows = []
for label, run in RUNS:
    for m in MODELS:
        f = Path("notebooks/rda-experiments/runs") / run / f"sample{SAMPLE}.rda.{m}.json"
        if not f.exists():
            print(f"{run}: no {m} output for sample {SAMPLE} - left out")
            continue
        judged, met, outcomes = E.evaluate_file(f, ref)
        c = Counter(outcomes.values())
        rows.append({"version": label.replace("\n", " "), "model": m, "run": run,
                     **{o: c[o] for o in OUTCOMES}, "F1": met["f1"],
                     "fields written": met["fields output"],
                     "not in reference": sum(r["reason"] == "not in the reference" for r in judged)})
        for field, o in outcomes.items():
            rows[-1].setdefault("_parts", []).append((E.row_of(field)[1], o))

df = pd.DataFrame([{k: v for k, v in r.items() if k != "_parts"} for r in rows])
n_fields = len(outcomes)
versions = [v for v in dict.fromkeys(df["version"])]
print(f"sample {SAMPLE}: {n_fields} annotated fields, {len(versions)} versions, {df['model'].nunique()} models")

totals = (df.groupby("version")[OUTCOMES].sum().reindex(versions))
totals["F1 (mean of models)"] = df.groupby("version")["F1"].mean().reindex(versions).round(3)
display(totals)
'''),

md("s2", '''
## 2. The figure: each version, per model and in total

Stacked bars - green correct, red mismatch, grey missed - add up to the annotated
fields (per model) or three times that (total). Saved next to the runs as
`version_comparison_sample{SAMPLE}.png`.
'''),

code("figure", r'''
panels = [(m, df[df["model"] == m].set_index("version").reindex(versions)) for m in MODELS]
panels.append(("total", totals))

fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6), sharey=False)
for ax, (name, d) in zip(axes, panels):
    bottom = np.zeros(len(d))
    for o in OUTCOMES:
        vals = d[o].fillna(0).to_numpy()
        ax.bar(range(len(d)), vals, bottom=bottom, color=COLOUR[o], width=0.72, label=o)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v:
                ax.text(i, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=10.5,
                        color="white" if o != "Missed" or v > 3 else INK, fontweight="bold")
        bottom += vals
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([l for l, _ in RUNS if l.replace("\n", " ") in d.index], fontsize=10)
    ax.set_title(NAME.get(name, f"All three models ({3 * n_fields} fields)"), fontsize=12.5,
                 fontweight="bold", loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, fontsize=10)
fig.suptitle(f"Prompt versions on sample {SAMPLE}: what each did with the {n_fields} annotated fields",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
out = OUT_DIR / f"version_comparison_sample{SAMPLE}.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print(f"saved -> {out}")
'''),

md("s2b", '''
## 2b. The table as a figure

The same numbers as a picture, for slides or a quick check: per model and in total,
correct / mismatch / missed (tinted like the field matrix - the darker the cell, the
larger its share of the annotated fields), then the mean F1 and how many fields the
version wrote that the reference does not have. Saved as
`version_comparison_table_sample{SAMPLE}.png`.
'''),

code("table_figure", r'''
import matplotlib.colors as mcolors

per_model = {m: df[df["model"] == m].set_index("version").reindex(versions) for m in MODELS}
extras = df.groupby("version")["not in reference"].sum().reindex(versions)

groups = [(NAME[m], per_model[m]) for m in MODELS] + [("All three models", totals)]
cols = []                                   # (group, outcome) or (group, None) spacer
for k, (g, _) in enumerate(groups):
    cols += [(g, o) for o in OUTCOMES] + ([(None, None)] if k < len(groups) - 1 else [])
cols += [(None, None), ("", "F1"), ("", "extras")]
surface = np.array(mcolors.to_rgb(SURFACE))


def cell(v, g, o):
    """(value, share of the annotated fields) for one cell."""
    d = dict(groups)[g]
    n = d.loc[v, o]
    denom = n_fields * (3 if g == "All three models" else 1)
    return int(n), n / denom


img = np.ones((len(versions), len(cols), 3)) * surface
fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 2.5, 0.62 * len(versions) + 2.4))
for i, v in enumerate(versions):
    for j, (g, o) in enumerate(cols):
        if g is None:
            continue
        if o in OUTCOMES:
            n, share = cell(v, g, o)
            strength = 0.12 + 0.88 * share if n else 0.04
            img[i, j] = surface * (1 - strength) + np.array(mcolors.to_rgb(COLOUR[o])) * strength
ax.imshow(img, aspect="auto")
for i, v in enumerate(versions):
    for j, (g, o) in enumerate(cols):
        if g is None:
            continue
        if o in OUTCOMES:
            n, share = cell(v, g, o)
            ax.text(j, i, str(n) if n else "–", ha="center", va="center", fontsize=12.5,
                    fontweight="bold", color="white" if share > 0.5 else INK)
        elif o == "F1":
            ax.text(j, i, f"{totals.loc[v, 'F1 (mean of models)']:.3f}", ha="center", va="center",
                    fontsize=11.5, color=INK)
        else:
            ax.text(j, i, f"{int(extras[v])}", ha="center", va="center", fontsize=11.5, color=MUTED)
ax.set_yticks(range(len(versions)))
ax.set_yticklabels(versions, fontsize=13, fontweight="bold")
ax.set_xticks([j for j, (g, o) in enumerate(cols) if g is not None])
ax.set_xticklabels([o if o in OUTCOMES else ("F1\nmean" if o == "F1" else "fields not\nin reference")
                    for g, o in cols if g is not None], fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for i in range(len(versions) + 1):
    ax.axhline(i - 0.5, color=SURFACE, linewidth=3, zorder=3)
for j in range(len(cols) + 1):
    ax.axvline(j - 0.5, color=SURFACE, linewidth=3, zorder=3)
for g, _ in groups:                         # group names above their three columns
    js = [j for j, (gg, o) in enumerate(cols) if gg == g]
    ax.annotate(g, xy=((js[0] + js[-1]) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 30),
                textcoords="offset points", ha="center", va="bottom", fontsize=12, fontweight="bold")
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"Prompt versions on sample {SAMPLE}: correct / mismatch / missed of the {n_fields} annotated fields",
             loc="left", fontsize=14, fontweight="bold", pad=62)
out = OUT_DIR / f"version_comparison_table_sample{SAMPLE}.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print(f"saved -> {out}")
'''),

md("s2c", '''
## 2c. The summary table

One row per version: correct / mismatch / missed summed over the three models, the mean
F1, and the prompt size (tokens sent to llama3.1 for this sample, from the run's
`run.json`), noting when the schema text is part of the prompt. The best value in each
score column is bold. Saved as `version_comparison_summary_sample{SAMPLE}.png`.
'''),

code("table_summary", r'''
import json as _json
import matplotlib.colors as mcolors


def prompt_size(run):
    folder = Path("notebooks/rda-experiments/runs") / run
    info = _json.loads((folder / "run.json").read_text(encoding="utf-8"))
    sent = next((t["tokens_sent"] for k, t in info["timings"].items()
                 if k.startswith("llama3.1:8b") and f"sample{SAMPLE}" in k), None)
    prompt = (folder / "prompt.txt").read_text(encoding="utf-8")
    with_schema = "{schema_text}" in prompt or "maDMP JSON schema" in prompt
    size = f"~{round(sent, -2):,} tokens" if sent else "?"
    return size + (" (schema text in prompt)" if with_schema else "")


summary = pd.DataFrame({
    "Version": versions,
    "Correct ↑": totals["Correct"].astype(int).to_numpy(),
    "Mismatch ↓": totals["Mismatch"].astype(int).to_numpy(),
    "Missed ↓": totals["Missed"].astype(int).to_numpy(),
    "F1 (mean)": totals["F1 (mean of models)"].map("{:.3f}".format).to_numpy(),
    "Prompt size": [prompt_size(dict((l.replace("\n", " "), r) for l, r in RUNS)[v]) for v in versions],
})
display(summary.set_index("Version"))

best = {"Correct ↑": totals["Correct"].idxmax(), "Mismatch ↓": totals["Mismatch"].idxmin(),
        "Missed ↓": totals["Missed"].idxmin(), "F1 (mean)": totals["F1 (mean of models)"].idxmax()}

# A heatmap: the score cells tinted like the field matrix (darker = larger share of the
# annotated fields), each showing the count and its share; F1 tinted blue; prompt size as text.
score_cols = [("Correct ↑", "Correct"), ("Mismatch ↓", "Mismatch"), ("Missed ↓", "Missed")]
cols = score_cols + [("F1 (mean)", None), ("Prompt size", None)]
n_total = 3 * n_fields
f1 = totals["F1 (mean of models)"]
f1_lo, f1_hi = f1.min(), f1.max()
surface = np.array(mcolors.to_rgb(SURFACE))
blue = np.array(mcolors.to_rgb("#2a78d6"))

img = np.ones((len(versions), len(cols), 3)) * surface
for i, v in enumerate(versions):
    for j, (label, outcome) in enumerate(cols):
        if outcome:
            n = int(totals.loc[v, outcome])
            strength = 0.12 + 0.88 * (n / n_total) if n else 0.04
            img[i, j] = surface * (1 - strength) + np.array(mcolors.to_rgb(COLOUR[outcome])) * strength
        elif label == "F1 (mean)":
            strength = 0.08 + 0.5 * ((f1[v] - f1_lo) / (f1_hi - f1_lo) if f1_hi > f1_lo else 0.5)
            img[i, j] = surface * (1 - strength) + blue * strength

fig, ax = plt.subplots(figsize=(13.5, 0.75 * len(versions) + 1.9))
ax.imshow(img, aspect="auto")
for i, v in enumerate(versions):
    for j, (label, outcome) in enumerate(cols):
        if outcome:
            n = int(totals.loc[v, outcome])
            share = n / n_total
            ink = "white" if share > 0.5 else INK
            if n:
                ax.text(j - 0.07, i, str(n), ha="right", va="center", fontsize=14, fontweight="bold", color=ink)
                ax.text(j + 0.04, i, f"{share:.0%}", ha="left", va="center", fontsize=11, color=ink)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=14, color="#c9c8c3")
        elif label == "F1 (mean)":
            weight = "bold" if v == best["F1 (mean)"] else "normal"
            ax.text(j, i, f"{f1[v]:.3f}", ha="center", va="center", fontsize=13, fontweight=weight, color=INK)
        else:
            ax.text(j - 0.45, i, summary["Prompt size"][i], ha="left", va="center", fontsize=11, color=INK)
ax.set_yticks(range(len(versions)))
ax.set_yticklabels(versions, fontsize=14, fontweight="bold")
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([f"{label}\n(of {n_total} fields)" if outcome else label for label, outcome in cols],
                   fontsize=11.5)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for i in range(len(versions) + 1):
    ax.axhline(i - 0.5, color=SURFACE, linewidth=4, zorder=3)
for j in range(len(cols) + 1):
    ax.axvline(j - 0.5, color=SURFACE, linewidth=4, zorder=3)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"Prompt versions on sample {SAMPLE} - three models together: count and share of the "
             f"{n_total} annotated fields", loc="left", fontsize=13.5, fontweight="bold", pad=44)
out = OUT_DIR / f"version_comparison_summary_sample{SAMPLE}.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print(f"saved -> {out}")
'''),

md("s3", '''
## 3. Per model

Correct / mismatch / missed and F1 for every version and model. F1 is scored over
every field either side wrote, so it also penalises fields the reference does not
have ("not in reference").
'''),

code("per_model", r'''
wide = df.pivot(index="version", columns="model", values=OUTCOMES + ["F1"]).reindex(versions)
wide = wide.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
wide = wide[[(m, c) for m in MODELS for c in OUTCOMES + ["F1"] if (m, c) in wide.columns]]
display(wide)
display(df.pivot(index="version", columns="model", values="not in reference").reindex(versions)
          .rename_axis(columns="fields written that the reference does not have"))
'''),

md("s4", '''
## 4. By part of the form, summed over the three models

Which parts of the form each version gets right. Each cell: correct / mismatch / missed
out of the annotated fields in that part (times three models).
'''),

code("per_part", r'''
parts = Counter(p for r in rows if r["version"] == versions[0] for p, _ in r["_parts"])
order = [p[1] for p in E.ROWS if p[1] in parts]
table = {}
for v in versions:
    c = Counter((p, o) for r in rows if r["version"] == v for p, o in r["_parts"])
    table[v] = [f"{c[(p, 'Correct')]} / {c[(p, 'Mismatch')]} / {c[(p, 'Missed')]}" for p in order]
part_df = pd.DataFrame(table, index=[f"{p}  ({parts[p]})" for p in order])
part_df.index.name = "part of the form (annotated fields x 3 models)"
display(part_df)
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
