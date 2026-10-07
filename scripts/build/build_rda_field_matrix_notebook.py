"""Build notebooks/rda-field-matrix.ipynb.

For one saved RDA maDMP run and one sample: every field the person annotated, marked
Correct, Wrong or Missed for each model. Drawn like the narrative pipeline's confusion
matrices (comparison-matrix-*.ipynb): one panel per model, rows are the annotated field
types, columns the three outcomes, cells the share of that row. A table underneath
lists every annotated field with each model's verdict.

Fields the person left empty are not part of this view. Reads the run's saved
evaluation only; no model is called.

    python scripts/build/build_rda_field_matrix_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/rda-field-matrix.ipynb")


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
# Annotated fields: correct, wrong or missed

For one run and one sample, every field the person annotated is checked against what each
model wrote at the same place:

| Outcome | Meaning |
|---|---|
| **Correct** | the model's value matches the annotation |
| **Wrong** | the model wrote a different value there |
| **Missed** | the model wrote nothing there, or left it empty |

Fields the person left empty are not shown. The matrix has one panel per model; each row is
a kind of field, with the number annotated in brackets, and each cell is the share of that
row. The table underneath lists every annotated field.

**To look at another run or sample:** change `RUN` and `SAMPLE` in the next cell.
'''),

code("setup", r'''
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

RUN    = "v4-sample14"                           # a folder under data/output/rda/runs/
SAMPLE = 14
RESULT = Path("data/output/rda/runs") / RUN / "evaluation_results.xlsx"
CHART  = Path("data/output/rda/runs") / RUN / f"field_matrix_sample{SAMPLE}.png"
MODELS = ["llama3.1-8b", "gemma4-e4b", "llama3.3-70b"]
OUTCOMES = ["Correct", "Wrong", "Missed"]

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
    "axes.titlecolor": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 220)
'''),

md("s1", '''
## 1. One outcome per annotated field and model
'''),

code("outcomes", r'''
details = pd.read_excel(RESULT, sheet_name="details")
if "sample" in details:
    details = details[details["sample"] == SAMPLE]
details["model"] = details["model"].str.replace(":", "-")


def outcome(rows):
    """The evaluation writes one or two rows per annotated field; reduce them to one outcome."""
    if (rows.verdict == "Correct").any():
        return "Correct"
    wrote = rows[(rows.verdict == "Hallucinated") & (rows.reason != "empty where the reference has a value")]
    return "Wrong" if len(wrote) else "Missed"


def kind(path):
    """A readable name for a kind of field: dmp.dataset[3].distribution[0].host.title -> dataset > host > title"""
    p = re.sub(r"\[\d+\]", "", path.replace("dmp.", "", 1))
    return p.replace("distribution.", "").replace(".", " > ").replace("_", " ")


annotated = details[details["reference value"].notna()]
fields = (annotated.groupby(["field", "model"]).apply(outcome, include_groups=False)
          .rename("outcome").reset_index())
fields["kind"] = fields["field"].map(kind)
values = annotated.drop_duplicates("field").set_index("field")["reference value"]
wrote = (annotated[annotated["model value"].notna()]
         .drop_duplicates(["field", "model"]).set_index(["field", "model"])["model value"])

models = [m for m in MODELS if m in set(fields.model)]
n_fields = fields.field.nunique()
print(f"run {RUN}, sample {SAMPLE}: {n_fields} annotated fields, models: {', '.join(models)}")
display(fields.pivot_table(index="model", columns="outcome", values="field", aggfunc="count", fill_value=0)
        .reindex(index=models, columns=OUTCOMES, fill_value=0))
'''),

md("s2", '''
## 2. The matrix

Rows: kinds of annotated fields, most frequent first, with how many were annotated.
Columns: what each model did with them. A row's three cells add up to 100%.
'''),

code("matrix", r'''
order = fields.drop_duplicates("field")["kind"].value_counts()
kinds = list(order.index)

fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 3.2, 0.42 * len(kinds) + 1.8),
                         sharey=True, squeeze=False)
axes = axes[0]
for ax, m in zip(axes, models):
    counts = (fields[fields.model == m].groupby(["kind", "outcome"]).size().unstack(fill_value=0)
              .reindex(index=kinds, columns=OUTCOMES, fill_value=0))
    share = counts.div(counts.sum(axis=1).replace(0, 1), axis=0).to_numpy()
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(kinds)):
        for j in range(len(OUTCOMES)):
            v, n = share[i, j], counts.iloc[i, j]
            ax.text(j, i, f"{v:.0%} ({n})" if n else "0", ha="center", va="center", fontsize=8,
                    fontweight="bold" if j == 0 and n else "normal",
                    color="#b9b9b4" if n == 0 else ("white" if v > 0.55 else INK))
    total = counts.sum()
    ax.set_title(f"{m}\n{total['Correct']} correct · {total['Wrong']} wrong · {total['Missed']} missed",
                 pad=26, fontsize=11)
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels(OUTCOMES)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5, len(OUTCOMES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(kinds), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="x", length=0)
    ax.add_patch(plt.Rectangle((-.5, -.5), 1, len(kinds), fill=False, edgecolor=INK, linewidth=1.6, zorder=5))
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_yticks(range(len(kinds)))
axes[0].set_yticklabels([f"{k}  ({order[k]})" for k in kinds])
plt.tight_layout()
cbar = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02, label="share of the row")
cbar.outline.set_visible(False)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
fig.suptitle(f"Run {RUN}, sample {SAMPLE}: {n_fields} annotated fields", fontweight="bold", y=1.03)
fig.savefig(CHART, dpi=200, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print(f"saved -> {CHART}")
'''),

md("s3", '''
## 3. Every annotated field

One row per field the person annotated: the annotated value, each model's outcome, and,
when it was wrong, what the model wrote instead.
'''),

code("table", r'''
wide = fields.pivot(index="field", columns="model", values="outcome").reindex(columns=models)
wide.insert(0, "annotated value", values.reindex(wide.index).astype(str).str.slice(0, 60))
for m in models:
    wide[f"{m} wrote"] = [str(wrote.get((f, m), ""))[:45] if wide.loc[f, m] == "Wrong" else ""
                          for f in wide.index]
natural = lambda f: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", f)]
display(wide.loc[sorted(wide.index, key=natural)].reset_index())
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
