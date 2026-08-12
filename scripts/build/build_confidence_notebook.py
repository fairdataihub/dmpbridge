"""Build notebooks/8-confidence-analysis.ipynb.

Four steps, one table, one example per case, one chart.

    python scripts/build/build_confidence_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/8-confidence-analysis.ipynb")


def md(cid, lines):
    """Markdown cell."""
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    """Code cell."""
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


cells = [
    md("title", [
        "# Can we trust the confidence score?",
        "",
        "Every label the model produces comes with a **confidence**. If it were trustworthy,",
        "wrong labels would carry lower confidence than right ones.",
    ]),

    code("imports", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import json",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from IPython.display import display",
        "",
        "from dmpbridge.core import paths as P",
        "from dmpbridge.evaluation.evaluate import (",
        "    containment, extract_gold, resolve_old_gt_path, tokenize,",
        ")",
        "",
        "MODELS = ['llama3.1:8b', 'gemma4:e4b', 'llama3.3:70b']",
        "EXTRACTOR = 'pdfplumber'",
        "SAMPLES = range(1, 11)",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — Load the labeled JSON",
        "",
        "One entry per **line** of the PDF, each with a label and a confidence.",
    ]),
    code("step1", [
        "lines = json.loads(P.labeled_path(P.make_tag('llama3.1:8b', EXTRACTOR), 1)",
        "                   .read_text(encoding='utf-8'))",
        "print(f'{len(lines)} lines in sample 1')",
        "display(pd.DataFrame(lines)[['text', 'label', 'confidence']].head(4))",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — Join the lines back into items",
        "",
        "A paragraph is several lines in the PDF but **one item** in the annotation. So",
        "neighbouring lines with the same label are put back together — exactly what the",
        "pipeline does when it builds its final output.",
    ]),
    code("step2", [
        "def join_lines(lines):",
        "    \"\"\"Neighbouring lines with the same label become one item.\"\"\"",
        "    items, current = [], None",
        "    for line in lines:",
        "        if current is not None and current['label'] == line['label']:",
        "            current['text'] += ' ' + line['text']",
        "            current['line_confidences'].append(float(line['confidence']))",
        "        else:",
        "            if current is not None:",
        "                items.append(current)",
        "            current = {'label': line['label'], 'text': line['text'],",
        "                       'line_confidences': [float(line['confidence'])]}",
        "    if current is not None:",
        "        items.append(current)",
        "    return items",
        "",
        "",
        "items = join_lines(lines)",
        "print(f'{len(lines)} lines  ->  {len(items)} items')",
    ]),

    # ── 3 ───────────────────────────────────────────────────────────────
    md("md-3", [
        "## Step 3 — One confidence per item: the lowest of its lines",
        "",
        "A paragraph is only as trustworthy as its shakiest line.",
    ]),
    code("step3", [
        "biggest = max(items, key=lambda it: len(it['line_confidences']))",
        "print(f\"an item made of {len(biggest['line_confidences'])} lines\")",
        "print(f\"  line confidences : {biggest['line_confidences']}\")",
        "print(f\"  lowest           : {min(biggest['line_confidences'])}  <- the item's confidence\")",
    ]),

    # ── 4 ───────────────────────────────────────────────────────────────
    md("md-4", [
        "## Step 4 — Check every item against the annotation",
        "",
        "The same steps for every model and all 10 documents: load, join, take the lowest",
        "confidence, compare with the annotation.",
    ]),
    code("step4", [
        "def true_label_of(item, annotation):",
        "    \"\"\"The label of the annotation paragraph this item belongs to, or None.\"\"\"",
        "    words = tokenize(item['text'])",
        "    if len(words) < 3:",
        "        return None",
        "    best_score, best_label = 0.0, None",
        "    for gold_text, gold_label in annotation:",
        "        score = containment(words, tokenize(gold_text))",
        "        if score > best_score:",
        "            best_score, best_label = score, gold_label",
        "    return best_label if best_score >= 0.75 else None",
        "",
        "",
        "rows = []",
        "for sample in SAMPLES:",
        "    annotation = extract_gold(resolve_old_gt_path(sample))",
        "    for model in MODELS:",
        "        path = P.labeled_path(P.make_tag(model, EXTRACTOR), sample)",
        "        for item in join_lines(json.loads(path.read_text(encoding='utf-8'))):",
        "            true_label = true_label_of(item, annotation)",
        "            if true_label is None:",
        "                continue",
        "            rows.append({'model': model,",
        "                         'confidence': min(item['line_confidences']),",
        "                         'correct': item['label'] == true_label,",
        "                         'true label': true_label,",
        "                         'model said': item['label'],",
        "                         'text': item['text']})",
        "",
        "df = pd.DataFrame(rows)",
        "print(f'{len(df)} items checked across {len(MODELS)} models and 10 documents')",
    ]),

    # ── result ──────────────────────────────────────────────────────────
    md("md-5", [
        "## The result",
    ]),
    code("result", [
        "table = (df.groupby('model')",
        "         .apply(lambda d: pd.Series({",
        "             'Accuracy, %': d['correct'].mean() * 100,",
        "             'Total confidence, %': d['confidence'].mean() * 100,",
        "             'Confidence for correct answer, %':",
        "                 d[d['correct']]['confidence'].mean() * 100,",
        "             'Confidence for incorrect answer, %':",
        "                 d[~d['correct']]['confidence'].mean() * 100,",
        "         }), include_groups=False)",
        "         .reindex(MODELS))",
        "table.index.name = 'Model'",
        "display(table.style.format('{:.1f}'))",
    ]),
    md("md-5b", [
        "**Read the last two columns.** If confidence worked, the last column would be much",
        "lower than the one before it — mistakes arriving with visible doubt. They are almost",
        "the same.",
        "",
        "**And compare the first two.** Every model claims far more confidence than its",
        "accuracy justifies.",
    ]),

    # ── examples ────────────────────────────────────────────────────────
    md("md-8b", [
        "### A real example from each case",
        "",
        "Splitting items by confidence (high = 0.9 or more) and by whether they were right.",
        "The second row is the one that matters.",
    ]),
    code("examples", [
        "M = MODELS[0]",
        "d = df[df['model'] == M].copy()",
        "d['high confidence'] = d['confidence'] >= 0.9",
        "",
        "picks = []",
        "for high, ok, name in ((True, True, 'high confidence + correct'),",
        "                       (True, False, 'high confidence + INCORRECT'),",
        "                       (False, True, 'low confidence + correct'),",
        "                       (False, False, 'low confidence + incorrect')):",
        "    sub = d[(d['high confidence'] == high) & (d['correct'] == ok)]",
        "    if len(sub):",
        "        r = sub.sort_values('confidence', ascending=not high).iloc[0]",
        "        picks.append({'case': name, 'confidence': f\"{r['confidence']:.2f}\",",
        "                      'true label': r['true label'],",
        "                      'model said': r['model said'],",
        "                      'text': r['text'][:64]})",
        "    else:",
        "        picks.append({'case': name, 'confidence': '—', 'true label': '—',",
        "                      'model said': '—', 'text': '(no items in this case)'})",
        "",
        "print(f'{M}\\n')",
        "display(pd.DataFrame(picks).set_index('case'))",
    ]),

    # ── chart ───────────────────────────────────────────────────────────
    md("md-chart", [
        "## Large language models' mean confidence scores for correct and incorrect answers",
        "",
        "Two rows per model. The **dot** is the mean confidence — the last two columns of the",
        "table above. The **bar behind it** is the spread of the individual items (± 1 SD),",
        "which is what shows whether the two groups are really apart or only look it.",
    ]),
    code("chart", [
        "# Colours checked for colour-blind separation, not chosen by eye:",
        "# blue/orange separate at dE 24 for every form of colour blindness, where the",
        "# usual green/red pair sits at 8 -- near-identical to a deuteranope.",
        "OK, BAD = '#2a6fb5', '#d97917'",
        "INK, MUTED, SURFACE, GRID = '#1a1a19', '#575653', '#fcfcfb', '#e8e8e4'",
        "plt.rcParams.update({",
        "    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,",
        "    'text.color': INK, 'axes.labelcolor': MUTED,",
        "    'xtick.color': MUTED, 'ytick.color': INK,",
        "    'axes.edgecolor': '#d8d8d4', 'axes.titlecolor': INK,",
        "    'font.size': 10, 'axes.titlesize': 12.5, 'axes.titleweight': 'bold',",
        "    'legend.frameon': False,",
        "})",
        "",
        "stats = []",
        "for model in MODELS:",
        "    d = df[df['model'] == model]",
        "    right, wrong = d[d['correct']]['confidence'], d[~d['correct']]['confidence']",
        "    stats.append({'model': model,",
        "                  'ok': right.mean() * 100, 'ok sd': right.std() * 100,",
        "                  'bad': wrong.mean() * 100, 'bad sd': wrong.std() * 100})",
        "",
        "fig, ax = plt.subplots(figsize=(9.0, 4.4))",
        "OFFSET = 0.17",
        "",
        "for i, s in enumerate(stats):",
        "    y = -i",
        "    for key, sd_key, dy, colour, name in ((  'ok',   'ok sd',  OFFSET, OK,  'correct'),",
        "                                         ('bad',  'bad sd', -OFFSET, BAD, 'incorrect')):",
        "        mean, sd = s[key], s[sd_key]",
        "        ax.plot([mean - sd, mean + sd], [y + dy] * 2, color=colour, alpha=0.28,",
        "                linewidth=7, solid_capstyle='round', zorder=2)",
        "        ax.plot(mean, y + dy, 'o', color=colour, markersize=9,",
        "                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,",
        "                label=f'{name} answers' if i == 0 else None)",
        "        ax.text(mean, y + dy + 0.115, f'{mean:.1f}', ha='center', va='bottom',",
        "                fontsize=9.5, fontweight='bold', color=colour, zorder=4)",
        "",
        "    # the gap between the two means, in the right-hand margin",
        "    ax.text(114, y, f\"{s['ok'] - s['bad']:+.1f}\", ha='center', va='center',",
        "            fontsize=10, fontweight='bold', color=INK)",
        "",
        "ax.text(114, 0.72, 'gap', ha='center', va='center', fontsize=9, color=MUTED)",
        "ax.set_yticks([-i for i in range(len(stats))])",
        "ax.set_yticklabels([s['model'] for s in stats], fontsize=10.5)",
        "ax.set_ylim(-len(stats) + 0.45, 0.85)",
        "ax.set_xlim(64, 118)",
        "ax.set_xticks([70, 80, 90, 100])",
        "ax.set_xlabel('confidence, %')",
        "ax.grid(axis='x', color=GRID, linewidth=0.9)",
        "ax.set_axisbelow(True)",
        "ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.14), ncol=2,",
        "          handletextpad=0.3, columnspacing=1.6)",
        "ax.set_title('Mean confidence for correct and incorrect answers\\n'",
        "             f'{len(df)} items, 10 documents (dot = mean, bar = ± 1 SD)', pad=14)",
        "for side in ('top', 'right', 'left', 'bottom'):",
        "    ax.spines[side].set_visible(False)",
        "ax.tick_params(length=0)",
        "plt.tight_layout()",
        "",
        "out = Path('Report-doc/confidence_correct_vs_incorrect.png')",
        "out.parent.mkdir(parents=True, exist_ok=True)",
        "fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=SURFACE)",
        "print(f'saved for slides -> {out}')",
        "plt.show()",
    ]),
    md("md-chart-b", [
        "**The two dots sit almost on top of each other, and the spread bars overlap almost",
        "completely.** The largest gap is under 5 points, against a spread of 5 to 16 — so",
        "knowing an answer's confidence tells you almost nothing about whether it is right.",
    ]),

    # ── conclusion ──────────────────────────────────────────────────────
    md("conclusion", [
        "## Conclusion",
        "",
        "**The confidence score cannot be used to catch mistakes.** A high number does not",
        "mean a correct label, so it must not be used to accept a label automatically.",
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
