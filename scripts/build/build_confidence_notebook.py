"""Build notebooks/8-confidence-analysis.ipynb — one step per cell, code visible.

Each step does exactly one thing and prints what it produced, so the whole
method can be read off the notebook without reading any other file.

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
        "Every label the model produces comes with a **confidence**. If that number were",
        "trustworthy, wrong labels would carry low confidence and we could catch mistakes",
        "automatically.",
        "",
        "One step per cell below, each showing its own code and result.",
    ]),

    code("imports", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import json",
        "import pandas as pd",
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
        "",
        "pd.set_option('display.max_colwidth', 60)",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — Load the labeled JSON",
        "",
        "This is what the model produced: one entry per **line** of the PDF, each with a",
        "label and a confidence.",
    ]),
    code("step1", [
        "path = P.labeled_path(P.make_tag('llama3.1:8b', EXTRACTOR), 1)",
        "print(path)",
        "",
        "lines = json.loads(path.read_text(encoding='utf-8'))",
        "print(f'{len(lines)} lines in this file')",
    ]),
    code("step1b", [
        "# The first four lines, showing only the fields this notebook uses.",
        "display(pd.DataFrame(lines)[['text', 'label', 'confidence']].head(4))",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — Join the lines back into items",
        "",
        "A paragraph in the PDF is several lines, but the annotation records it as **one",
        "item**. So neighbouring lines carrying the **same label** are put back together.",
        "",
        "This is what the pipeline already does when it builds its final output.",
    ]),
    code("step2", [
        "def join_lines(lines):",
        "    \"\"\"Neighbouring lines with the same label become one item.\"\"\"",
        "    items = []",
        "    current = None",
        "    for line in lines:",
        "        label = line['label']",
        "        confidence = float(line['confidence'])",
        "        if current is not None and current['label'] == label:",
        "            current['text'] += ' ' + line['text']",
        "            current['line_confidences'].append(confidence)",
        "        else:",
        "            if current is not None:",
        "                items.append(current)",
        "            current = {'label': label,",
        "                       'text': line['text'],",
        "                       'line_confidences': [confidence]}",
        "    if current is not None:",
        "        items.append(current)",
        "    return items",
    ]),
    code("step2b", [
        "items = join_lines(lines)",
        "print(f'{len(lines)} lines  ->  {len(items)} items')",
    ]),
    md("md-2b", [
        "### See it happen on one paragraph",
        "",
        "Lines 3, 4 and 5 of the file all carry the same label, so they become a single item.",
    ]),
    code("step2c", [
        "print('BEFORE — three separate lines:')",
        "for line in lines[3:6]:",
        "    print(f\"  [{line['label']}]  conf {line['confidence']}  {line['text'][:58]!r}\")",
        "",
        "print()",
        "print('AFTER — one item:')",
        "joined = join_lines(lines[3:6])[0]",
        "print(f\"  [{joined['label']}]  from {len(joined['line_confidences'])} lines\")",
        "print(f\"  {joined['text'][:110]!r}\")",
    ]),

    # ── 3 ───────────────────────────────────────────────────────────────
    md("md-3", [
        "## Step 3 — Give each item one confidence",
        "",
        "An item is built from several lines, each with its own confidence. We take the",
        "**lowest** — a paragraph is only as trustworthy as its shakiest line.",
    ]),
    code("step3", [
        "for item in items:",
        "    item['confidence'] = min(item['line_confidences'])",
        "",
        "# The item built from the most lines, to show the rule working.",
        "biggest = max(items, key=lambda it: len(it['line_confidences']))",
        "print(f\"an item made of {len(biggest['line_confidences'])} lines\")",
        "print(f\"  line confidences : {biggest['line_confidences']}\")",
        "print(f\"  lowest           : {biggest['confidence']}  <- the item's confidence\")",
    ]),

    # ── 4 ───────────────────────────────────────────────────────────────
    md("md-4", [
        "## Step 4 — Check each item against the annotation",
        "",
        "The annotation says what is really in the document. Each item is matched to the",
        "paragraph it belongs to, then we check whether the label agrees.",
    ]),
    code("step4", [
        "def check(item, annotation):",
        "    \"\"\"Return the item's true label, or None if it matches no paragraph.\"\"\"",
        "    words = tokenize(item['text'])",
        "    if len(words) < 3:",
        "        return None                     # too short to place reliably",
        "    best_score, best_label = 0.0, None",
        "    for gold_text, gold_label in annotation:",
        "        score = containment(words, tokenize(gold_text))",
        "        if score > best_score:",
        "            best_score, best_label = score, gold_label",
        "    return best_label if best_score >= 0.75 else None",
    ]),
    code("step4b", [
        "annotation = extract_gold(resolve_old_gt_path(1))",
        "print(f'the annotation for sample 1 has {len(annotation)} items\\n')",
        "",
        "for item in items[:6]:",
        "    true_label = check(item, annotation)",
        "    if true_label is None:",
        "        print(f\"  (skipped — no match)  {item['text'][:52]!r}\")",
        "    else:",
        "        verdict = 'correct' if item['label'] == true_label else 'WRONG'",
        "        print(f\"  conf {item['confidence']:.2f}  said {item['label']:<20}\"",
        "              f\"true {true_label:<20}{verdict}\")",
    ]),

    # ── 5 ───────────────────────────────────────────────────────────────
    md("md-5", [
        "## Step 5 — Do all of that for every model and every document",
        "",
        "The same four steps, in a loop: load, join, take the lowest confidence, check.",
    ]),
    code("step5", [
        "rows = []",
        "for sample in SAMPLES:",
        "    annotation = extract_gold(resolve_old_gt_path(sample))",
        "    for model in MODELS:",
        "        path = P.labeled_path(P.make_tag(model, EXTRACTOR), sample)",
        "        lines = json.loads(path.read_text(encoding='utf-8'))",
        "        for item in join_lines(lines):",
        "            item['confidence'] = min(item['line_confidences'])",
        "            true_label = check(item, annotation)",
        "            if true_label is None:",
        "                continue",
        "            rows.append({'model': model,",
        "                         'sample': sample,",
        "                         'confidence': item['confidence'],",
        "                         'model said': item['label'],",
        "                         'true label': true_label,",
        "                         'correct': item['label'] == true_label,",
        "                         'text': item['text']})",
        "",
        "df = pd.DataFrame(rows)",
        "print(f'{len(df)} items checked')",
        "display(df.head(5))",
    ]),

    # ── 6 ───────────────────────────────────────────────────────────────
    md("md-6", [
        "## Step 6 — The answer",
        "",
        "If confidence were trustworthy, the wrong items would be clearly less confident",
        "than the right ones.",
    ]),
    code("step6", [
        "answer = (df.groupby('model')",
        "          .apply(lambda d: pd.Series({",
        "              'items': len(d),",
        "              'correct': d['correct'].sum(),",
        "              'accuracy': d['correct'].mean(),",
        "              'confidence when RIGHT': d[d['correct']]['confidence'].mean(),",
        "              'confidence when WRONG': d[~d['correct']]['confidence'].mean(),",
        "          }), include_groups=False)",
        "          .reindex(MODELS))",
        "display(answer.style.format({'items': '{:.0f}', 'correct': '{:.0f}',",
        "                             'accuracy': '{:.0%}',",
        "                             'confidence when RIGHT': '{:.2f}',",
        "                             'confidence when WRONG': '{:.2f}'}))",
    ]),
    md("md-6b", [
        "### The wrong items that were most confident",
    ]),
    code("step6b", [
        "worst = (df[~df['correct']].sort_values('confidence', ascending=False)",
        "         [['model', 'confidence', 'true label', 'model said', 'text']]",
        "         .head(8).reset_index(drop=True))",
        "display(worst.style.background_gradient(cmap='Reds', subset=['confidence'],",
        "                                        vmin=0.5, vmax=1.0)",
        "        .format({'confidence': '{:.2f}'}))",
    ]),

    # ── conclusion ──────────────────────────────────────────────────────
    md("conclusion", [
        "## Conclusion",
        "",
        "**The confidence score cannot be used to catch mistakes.**",
        "",
        "The two confidence columns in step 6 are close together for every model — wrong",
        "items are about as confident as right ones. The table above shows mistakes claimed",
        "at complete certainty.",
        "",
        "So a high confidence does not mean a correct label, and it must not be used to",
        "accept one automatically.",
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
