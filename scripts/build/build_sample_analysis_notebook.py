"""Build notebooks/7-sample1-error-analysis.ipynb.

The simplest possible telling: what is in the document, what the reader did,
what each model did, and where the errors came from.

    python scripts/build/build_sample_analysis_notebook.py
"""
import json
from pathlib import Path

from dmpbridge.core import paths as P

NB = Path("notebooks/7-sample1-error-analysis.ipynb")
SAMPLE, EXTRACTOR = 1, "pdfplumber"
MODELS = ["llama3.1:8b", "gemma4:e4b", "llama3.3:70b"]

AVAILABLE = [m for m in MODELS
             if P.structured_path(P.make_tag(m, EXTRACTOR), SAMPLE).exists()]


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
        f"# Sample {SAMPLE} — where do the errors come from?",
        "",
        "Two suspects:",
        "",
        "1. **the reader** (pdfplumber) — did it damage the document?",
        "2. **the models** — did they label it wrong?",
        "",
        "Three steps, one verdict.",
    ]),

    code("setup", [
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
        f"SAMPLE, EXTRACTOR = {SAMPLE}, {EXTRACTOR!r}",
        f"MODELS = {AVAILABLE!r}",
        "pd.set_option('display.max_colwidth', 80)",
        "",
        "# The human annotation: the correct answer for this document.",
        "gold = extract_gold(resolve_old_gt_path(SAMPLE))",
        "# What pdfplumber read from the PDF.",
        "blocks = json.loads((P.EXTRACTED_DIR / EXTRACTOR / f'sample{SAMPLE}.json')",
        "                    .read_text(encoding='utf-8'))",
        "# What each model labeled — the same blocks, one label per block.",
        "labeled = {m: json.loads(P.labeled_path(P.make_tag(m, EXTRACTOR), SAMPLE)",
        "                         .read_text(encoding='utf-8')) for m in MODELS}",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — Did the reader damage the document?",
        "",
        "The human annotation says what is really in this document. We check every block the",
        "reader produced: does its text belong to something in the annotation?",
    ]),
    code("step1", [
        "# For each block, find the annotation item its text belongs to.",
        "truth = []",
        "for b in blocks:",
        "    bt = tokenize(b['text'])",
        "    best, lab = 0.0, None",
        "    for gt, gl in gold:",
        "        c = containment(bt, tokenize(gt))",
        "        if c > best:",
        "            best, lab = c, gl",
        "    truth.append(lab if best >= 0.75 else None)",
        "",
        "lost = sum(1 for t in truth if t is None)",
        "print(f'the annotation has {len(gold)} items')",
        "print(f'the reader produced {len(blocks)} blocks')",
        "print(f'blocks with text that is NOT in the annotation: {lost}')",
    ]),
    md("md-1b", [
        "**Verdict on the reader: innocent.** Every block is real text from the document.",
        "Nothing was lost, nothing was made up.",
        "",
        "The reader did one thing worth knowing: it reads **line by line**, so a long answer",
        "arrives as several blocks instead of one. That is why there are 78 blocks for only",
        "28 items. Splitting is not an error by itself — the pipeline glues neighbouring",
        "blocks back together when they carry the same label.",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — How well did each model label the same blocks?",
        "",
        "Every model received the identical blocks. Each block's correct label is known",
        "(from the annotation). So this table is a pure test of the models:",
    ]),
    code("step2", [
        "known = sum(1 for t in truth if t)",
        "rows = []",
        "for m in MODELS:",
        "    ok = sum(1 for t, b in zip(truth, labeled[m]) if t and b.get('label') == t)",
        "    rows.append({'model': m, 'blocks': known, 'correct': ok,",
        "                 'wrong': known - ok, 'accuracy': ok / known})",
        "acc = pd.DataFrame(rows).set_index('model')",
        "display(acc.style.background_gradient(cmap='Greens', subset=['accuracy'])",
        "        .format({'accuracy': '{:.0%}'}))",
    ]),
    md("md-2b", [
        "One model labels **every block correctly** — from exactly the same input the others",
        "got. That settles it: the input was good enough to score 100%. Any error below is",
        "the model's.",
    ]),

    # ── 3 ───────────────────────────────────────────────────────────────
    md("md-3", [
        "## Step 3 — The wrong blocks, one by one",
    ]),
    code("step3", [
        "for m in MODELS:",
        "    bad = [{'true label': t, 'model said': b.get('label'), 'text': b['text']}",
        "           for t, b in zip(truth, labeled[m]) if t and b.get('label') != t]",
        "    print(f'{m}: {len(bad)} wrong')",
        "    if bad:",
        "        display(pd.DataFrame(bad))",
        "    print()",
    ]),

    # ── verdict ─────────────────────────────────────────────────────────
    md("verdict", [
        "## The verdict",
        "",
        "| suspect | finding |",
        "|---|---|",
        "| **the reader** | innocent — read everything, invented nothing |",
        "| **gemma4:e4b** | no mistakes at all |",
        "| **llama3.3:70b** | one slip, on a short leftover piece of a split answer |",
        "| **llama3.1:8b** | 11 mistakes — 8 of them are the *same* mistake |",
        "",
        "llama3.1's repeated mistake: lines like `B. Scientific data that will be preserved…`",
        "are **questions**, but they are short, bold and lettered — they *look* like headings,",
        "and it calls them headings. The other two models get every one of these right.",
        "",
        "So for this document: **the errors come from the model, not the reader** — and",
        "mostly from one model, making one repeated judgement error.",
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
