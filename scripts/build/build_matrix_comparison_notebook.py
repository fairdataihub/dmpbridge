"""Build notebooks/comparison-matrix-pdfplumber-vs-lightonocr.ipynb.

Just the confusion matrix, side by side — pdfplumber+gemma vs LightOnOCR+gemma,
Path B only, percentages. No headline table, no per-class chart, no
per-sample breakdown — those already live in
comparison-gemma-pdfplumber-vs-lightonocr.ipynb. This is the focused,
one-picture version.

    python scripts/build/build_matrix_comparison_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/comparison-matrix-pdfplumber-vs-lightonocr.ipynb")
MODEL = "gemma4:e4b"


def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = [
    md("title", [
        f"# Confusion matrix — pdfplumber vs LightOnOCR, {MODEL}",
        "",
        "Path B only (the final, rules-applied output), one matrix per extractor, same model.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "from dmpbridge.core import paths as P",
        "from dmpbridge.evaluation.evaluate import (",
        "    LABELS, _confusion_from_match, _match_structured, extract_gold,",
        ")",
        "from dmpbridge.evaluation.annotation_rules import resolve_new_gt_path",
        "",
        f"MODEL = {MODEL!r}",
        "SAMPLES = range(1, 11)",
        "PANELS = {",
        "    'Pdfplumber+Gemma':  P.make_tag(MODEL, 'pdfplumber'),",
        f"    'LightOnOCR+Gemma':  '{MODEL.replace(':', '-').replace('.', '')}_lightonocr_whole_doc',",
        "}",
        "",
        "INK, SURFACE = '#0b0b0b', '#fcfcfb'",
        "plt.rcParams.update({",
        "    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,",
        "    'text.color': INK, 'axes.titlecolor': INK, 'font.size': 10,",
        "    'axes.titlesize': 12, 'axes.titleweight': 'bold',",
        "})",
        "",
        "CONF, available = {}, []",
        "for name, tag in PANELS.items():",
        "    if not all(P.final_path(tag, n).exists() for n in SAMPLES):",
        "        print(f'not available, skipped: {name}')",
        "        continue",
        "    from collections import defaultdict",
        "    pooled = defaultdict(lambda: defaultdict(int))",
        "    for n in SAMPLES:",
        "        gold = extract_gold(resolve_new_gt_path(n), dedup_question_title=False)",
        "        records, no_gold = _match_structured(P.final_path(tag, n), gold,",
        "                                             dedup_question_title=False)",
        "        this = _confusion_from_match(records, no_gold)",
        "        for k, row in this.items():",
        "            for kk, v in row.items():",
        "                pooled[k][kk] += v",
        "    CONF[name] = pooled",
        "    available.append(name)",
        "",
        "print('comparing:', ', '.join(available))",
    ]),

    code("matrix", [
        "COLS = list(LABELS) + ['(not labeled)']",
        "KEYS = list(LABELS) + ['__missed__']",
        "",
        "",
        "def build_pct(conf):",
        "    pct = np.zeros((len(LABELS), len(KEYS)))",
        "    for i, t in enumerate(LABELS):",
        "        row_total = sum(conf.get(t, {}).values())",
        "        for j, key in enumerate(KEYS):",
        "            if row_total:",
        "                pct[i, j] = conf.get(t, {}).get(key, 0) / row_total",
        "    return pct",
        "",
        "",
        "fig, axes = plt.subplots(1, len(available), figsize=(6.4 * len(available), 5.6),",
        "                         squeeze=False)",
        "axes = axes[0]",
        "",
        "for ax, name in zip(axes, available):",
        "    ax.set_title(name, pad=12)",
        "    cm = build_pct(CONF[name])",
        "    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)",
        "    for i in range(len(LABELS)):",
        "        for j in range(len(KEYS)):",
        "            v = cm[i, j]",
        "            ax.text(j, i, f'{v:.0%}' if v else '0', ha='center', va='center',",
        "                    fontsize=10, fontweight='bold' if i == j else 'normal',",
        "                    color='#b9b9b4' if v == 0 else ('white' if v > 0.55 else INK))",
        "    ax.set_xticks(range(len(COLS)))",
        "    ax.set_xticklabels(COLS, rotation=30, ha='right')",
        "    ax.set_yticks(range(len(LABELS)))",
        "    ax.set_yticklabels(LABELS)",
        "    ax.set_xlabel('Predicted label', labelpad=10)",
        "    ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True)",
        "    ax.set_yticks(np.arange(-.5, len(LABELS), 1), minor=True)",
        "    ax.grid(which='minor', color=SURFACE, linewidth=2)",
        "    ax.tick_params(which='minor', length=0)",
        "    for k in range(len(LABELS)):",
        "        ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False,",
        "                                   edgecolor=INK, linewidth=2.0, zorder=5))",
        "    ax.axvline(len(LABELS) - 0.5, color=INK, linewidth=2.5, zorder=6)",
        "    ax.get_xticklabels()[-1].set_style('italic')",
        "    for spine in ax.spines.values():",
        "        spine.set_visible(False)",
        "",
        "axes[0].set_ylabel('True label', labelpad=10)",
        "plt.tight_layout()",
        "",
        "# Colorbar and suptitle added after tight_layout, or it reserves no room for them.",
        "cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02,",
        "                    label='share of the true class')",
        "cbar.outline.set_visible(False)",
        "cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])",
        "cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])",
        f"fig.suptitle('{MODEL} \\u2014 confusion matrix, Path B', fontweight='bold', y=1.04)",
        "plt.show()",
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
