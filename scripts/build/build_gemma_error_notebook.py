"""Build notebooks/analysis-gemma4-e4b-errors.ipynb.

Two things, nothing else: a count confusion matrix (rows = true label,
columns = predicted label, raw counts — the same matrix format already used
in notebooks/results-gemma4-e4b-pdfplumber.ipynb, reused here so it's
self-contained) so it's obvious at a glance whether the pipeline is working,
and a flat table listing only the items that did NOT get labeled correctly
— nothing about items that matched.

    python scripts/build/build_gemma_error_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/analysis-gemma4-e4b-errors.ipynb")
MODEL = "gemma4:e4b"
TAG = "gemma4-e4b_pdfplumber_whole_doc"


def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = [
    md("title", [
        f"# {MODEL} — where the pipeline gets it wrong",
        "",
        "Two things, in order:",
        "",
        "1. **The confusion matrix** — every real item, sorted by what the pipeline actually",
        "   labeled it as. This is the fastest way to tell if the pipeline works: a matrix that's",
        "   almost all diagonal means it's labeling correctly; a matrix with a lot of weight",
        "   elsewhere means it's systematically confusing two particular label types.",
        "2. **Only the items that didn't get labeled correctly**, one row per item, across all 10",
        "   samples — no items that matched are shown.",
        "",
        "Both cover **Path B**: the pipeline's final output (after the annotation rules run),",
        "checked against the revised answer key.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from IPython.display import display",
        "",
        "from dmpbridge.core import paths as P",
        "from dmpbridge.evaluation.evaluate import LABELS, extract_gold, _match_structured",
        "from dmpbridge.evaluation.annotation_rules import (",
        "    load_method_new, convert_tag_to_final, resolve_new_gt_path,",
        ")",
        "",
        f"MODEL, TAG = {MODEL!r}, {TAG!r}",
        "SAMPLES = range(1, 11)",
        "",
        "pd.set_option('display.max_colwidth', None)   # never truncate the text column",
        "INK, SURFACE = '#0b0b0b', '#fcfcfb'",
        "",
        "df_b, conf_b, err_b = load_method_new(TAG, exclude=[])",
        "if df_b is None:",
        "    convert_tag_to_final(TAG)",
        "    df_b, conf_b, err_b = load_method_new(TAG, exclude=[])",
        "print(f'{MODEL}: {len(df_b)} documents scored')",
    ]),

    md("md-matrix", [
        "## 1. Confusion matrix — raw counts",
        "",
        "Rows are the true label, columns are what the pipeline actually produced.",
        "**The `(not labeled)` column is items the pipeline never produced anything for at",
        "all** — it isn't a label choice, it's the absence of one. Read across a row to see",
        "everything that happened to one true class; the diagonal cell is how many it got right.",
    ]),
    code("matrix", [
        "COLS = list(LABELS) + ['(not labeled)']",
        "KEYS = list(LABELS) + ['__missed__']",
        "",
        "counts = np.zeros((len(LABELS), len(KEYS)), dtype=int)",
        "for i, t in enumerate(LABELS):",
        "    for j, key in enumerate(KEYS):",
        "        counts[i, j] = conf_b.get(t, {}).get(key, 0)",
        "",
        "fig, ax = plt.subplots(figsize=(7.6, 5.8))",
        "im = ax.imshow(counts, cmap='Blues', vmin=0, vmax=counts.max())",
        "for i in range(len(LABELS)):",
        "    for j in range(len(KEYS)):",
        "        v = int(counts[i, j])",
        "        ax.text(j, i, str(v) if v else '0', ha='center', va='center', fontsize=10,",
        "                fontweight='bold' if i == j else 'normal',",
        "                color='#b9b9b4' if v == 0 else ('white' if v / counts.max() > 0.55 else INK))",
        "",
        "ax.set_xticks(range(len(COLS)))",
        "ax.set_xticklabels(COLS, rotation=30, ha='right')",
        "ax.set_yticks(range(len(LABELS)))",
        "ax.set_yticklabels(LABELS)",
        "ax.set_xlabel('Predicted label', labelpad=10)",
        "ax.set_ylabel('True label', labelpad=10)",
        "",
        "ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True)",
        "ax.set_yticks(np.arange(-.5, len(LABELS), 1), minor=True)",
        "ax.grid(which='minor', color=SURFACE, linewidth=2)",
        "ax.tick_params(which='minor', length=0)",
        "for k in range(len(LABELS)):",
        "    ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False,",
        "                               edgecolor=INK, linewidth=2.0, zorder=5))",
        "ax.axvline(len(LABELS) - 0.5, color=INK, linewidth=2.5, zorder=6)",
        "ax.get_xticklabels()[-1].set_style('italic')",
        "for spine in ax.spines.values():",
        "    spine.set_visible(False)",
        "",
        "fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='items')",
        "fig.suptitle(f'{MODEL} \\u2014 Path B, confusion matrix (raw counts)', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
    ]),

    md("md-errors", [
        "## 2. Items not labeled correctly",
        "",
        "One row per item, only where the pipeline's answer disagrees with the answer key —",
        "either it labeled the text as the wrong type, produced nothing for a real item, or",
        "produced an item the answer key doesn't have at all.",
        "",
        "**Only the first two of those three (`wrong label`, `missing`) are the errors the",
        "matrix above can show.** The matrix has one row per *real* (answer-key) item, so a",
        "wrong label shows up off the diagonal and a missing item shows up in `(not labeled)`.",
        "`extra — not in answer key` items have no real item to be a row of at all — they're",
        "the pipeline producing something nobody asked for — so they exist only in the table",
        "below, never in the matrix. That's the whole reason the two totals don't match.",
    ]),
    code("errors", [
        "rows = []",
        "for n in SAMPLES:",
        "    gold = extract_gold(resolve_new_gt_path(n), dedup_question_title=False)",
        "    records, no_gold = _match_structured(P.final_path(TAG, n), gold, dedup_question_title=False)",
        "    for r in records:",
        "        if r['pred_text'] is None:",
        "            rows.append({'sample': n, 'true label': r['gold_label'],",
        "                         'predicted label': '(not produced)', 'issue': 'missing',",
        "                         'text': r['gold_text']})",
        "        elif r['pred_label'] != r['gold_label']:",
        "            rows.append({'sample': n, 'true label': r['gold_label'],",
        "                         'predicted label': r['pred_label'], 'issue': 'wrong label',",
        "                         'text': r['pred_text']})",
        "    for text, label in no_gold:",
        "        rows.append({'sample': n, 'true label': '(none)', 'predicted label': label,",
        "                     'issue': 'extra \\u2014 not in answer key', 'text': text})",
        "",
        "errors = pd.DataFrame(rows, columns=['sample', 'true label', 'predicted label', 'issue', 'text'])",
        "total_gold = sum(len(extract_gold(resolve_new_gt_path(n), dedup_question_title=False)) for n in SAMPLES)",
        "in_matrix = (errors['issue'] != 'extra \\u2014 not in answer key').sum()",
        "not_in_matrix = (errors['issue'] == 'extra \\u2014 not in answer key').sum()",
        "correct = total_gold - in_matrix",
        "",
        "print(f'{total_gold} real (answer-key) items total \\u2014 this is what the matrix rows sum to')",
        "print(f'  {correct} correct (the matrix diagonal)')",
        "print(f'  {in_matrix} wrong or missing (visible in the matrix, off-diagonal / \"(not labeled)\")')",
        "print(f'  + {not_in_matrix} extra items with no real counterpart (NOT in the matrix at all)')",
        "print(f'  = {len(errors)} rows in the table below')",
        "print()",
        "display(errors.groupby(['sample', 'issue']).size().unstack(fill_value=0))",
    ]),
    code("errors-table", [
        "display(errors)",
    ]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
