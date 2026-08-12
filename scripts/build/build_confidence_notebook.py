"""Build notebooks/8-confidence-analysis.ipynb.

One question, four steps, one table.

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
        "The same four steps for every model and all 10 documents: load, join, take the",
        "lowest confidence, compare with the annotation.",
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
        "                         'correct': item['label'] == true_label})",
        "",
        "df = pd.DataFrame(rows)",
        "print(f'{len(df)} items checked across {len(MODELS)} models and 10 documents')",
    ]),

    # ── the answer ──────────────────────────────────────────────────────
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
        "lower than the one before it — mistakes arriving with visible doubt.",
        "",
        "They are almost the same. For llama3.1:8b they differ by less than half a point: it",
        "is as confident when wrong as when right.",
        "",
        "**And compare the first two columns.** Every model claims far more confidence than",
        "its accuracy justifies — llama3.1:8b is right 49% of the time while claiming 89%.",
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
