"""Build notebooks/7-error-analysis.ipynb.

The same three-step investigation run on two documents that reach opposite
verdicts: sample 1 (the model's fault) and sample 6 (the reader's fault).

    python scripts/build/build_sample_analysis_notebook.py
"""
import json
from pathlib import Path

from dmpbridge.core import paths as P

NB = Path("notebooks/7-error-analysis.ipynb")
EXTRACTOR = "pdfplumber"
MODELS = ["llama3.1:8b", "gemma4:e4b", "llama3.3:70b"]

AVAILABLE = [m for m in MODELS
             if P.structured_path(P.make_tag(m, EXTRACTOR), 1).exists()]


def md(cid, lines):
    """Markdown cell."""
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    """Code cell."""
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def sample_cells(n):
    """The three investigation steps for one sample. Conclusions are appended
    separately, since the two documents reach different verdicts."""
    return [
        md(f"s{n}-md-1", [
            f"### Step 1 — Did the reader damage sample {n}?",
            "",
            "Every block the reader produced is checked against the annotation: does its",
            "text belong to **one** real item?",
        ]),
        code(f"s{n}-step1", [
            f"gold, blocks, truth = load({n})",
            "",
            "lost = [i for i, t in enumerate(truth) if t is None]",
            "print(f'the annotation has {len(gold)} items')",
            "print(f'the reader produced {len(blocks)} blocks')",
            "print(f'blocks whose text does not fit any single item: {len(lost)}')",
            "for i in lost:",
            "    print(f'   block {i:>2}: {blocks[i][\"text\"][:66]!r}')",
        ]),
        md(f"s{n}-md-2", [
            "### Step 2 — How well did each model label the same blocks?",
        ]),
        code(f"s{n}-step2", [
            f"labeled = {{m: load_labels(m, {n}) for m in MODELS}}",
            "known = sum(1 for t in truth if t)",
            "acc = pd.DataFrame([",
            "    {'model': m,",
            "     'correct': sum(1 for t, b in zip(truth, labeled[m]) if t and b.get('label') == t),",
            "     'wrong': sum(1 for t, b in zip(truth, labeled[m]) if t and b.get('label') != t),",
            "     'accuracy': sum(1 for t, b in zip(truth, labeled[m]) if t and b.get('label') == t) / known}",
            "    for m in MODELS]).set_index('model')",
            "display(acc.style.background_gradient(cmap='Greens', subset=['accuracy'])",
            "        .format({'accuracy': '{:.0%}'}))",
        ]),
        md(f"s{n}-md-3", [
            "### Step 3 — The wrong blocks, one by one",
        ]),
        code(f"s{n}-step3", [
            "for m in MODELS:",
            "    bad = [{'true label': t, 'model said': b.get('label'), 'text': b['text']}",
            "           for t, b in zip(truth, labeled[m]) if t and b.get('label') != t]",
            "    print(f'{m}: {len(bad)} wrong')",
            "    if bad:",
            "        display(pd.DataFrame(bad))",
            "    print()",
        ]),
    ]


cells = [
    md("title", [
        "# Where do the errors come from?",
        "",
        "Two suspects:",
        "",
        "1. **the reader** (pdfplumber) — did it damage the document?",
        "2. **the models** — did they label it wrong?",
        "",
        "The same three-step check is run on two documents — and they reach **opposite",
        "verdicts**.",
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
        f"EXTRACTOR = {EXTRACTOR!r}",
        f"MODELS = {AVAILABLE!r}",
        "pd.set_option('display.max_colwidth', 80)",
        "",
        "",
        "def load(n):",
        "    \"\"\"Annotation, reader blocks, and each block's true label for sample n.",
        "",
        "    A block's true label is the label of the annotation item its text belongs",
        "    to. A block that does not fit any single item gets None — that only",
        "    happens when one block spans two items at once.",
        "    \"\"\"",
        "    gold = extract_gold(resolve_old_gt_path(n))",
        "    blocks = json.loads((P.EXTRACTED_DIR / EXTRACTOR / f'sample{n}.json')",
        "                        .read_text(encoding='utf-8'))",
        "    truth = []",
        "    for b in blocks:",
        "        bt = tokenize(b['text'])",
        "        best, lab = 0.0, None",
        "        for gt, gl in gold:",
        "            c = containment(bt, tokenize(gt))",
        "            if c > best:",
        "                best, lab = c, gl",
        "        truth.append(lab if best >= 0.75 else None)",
        "    return gold, blocks, truth",
        "",
        "",
        "def load_labels(model, n):",
        "    \"\"\"The same blocks after one model labeled them.\"\"\"",
        "    return json.loads(P.labeled_path(P.make_tag(model, EXTRACTOR), n)",
        "                      .read_text(encoding='utf-8'))",
    ]),

    # ── sample 1 ────────────────────────────────────────────────────────
    md("s1-head", ["## Sample 1"]),
    *sample_cells(1),
    md("s1-conclusion", [
        "### Conclusion for sample 1",
        "",
        "**The errors come from the model, not from pdfplumber.**",
        "",
        "gemma4 received exactly the same blocks and labeled every one correctly — so the",
        "input was good enough to score 100%, and every error here belongs to the model.",
        "",
        "llama3.1's mistakes are mostly one repeated error: lettered lines like",
        "`B. Scientific data that will be preserved…` are questions, but they *look* like",
        "headings, and it calls them headings.",
    ]),

    # ── sample 6 ────────────────────────────────────────────────────────
    md("s6-head", [
        "## Sample 6 — the opposite case",
        "",
        "Same check, different document. This one's section headings are **underlined**",
        "instead of bold.",
    ]),
    *sample_cells(6),
    md("s6-conclusion", [
        "### Conclusion for sample 6",
        "",
        "**Here the reader is at fault — and no model can fix it.**",
        "",
        "This document's headings are underlined, and an underline is a drawn line the reader",
        "cannot see. So each heading arrives **glued to its answer in one block** — two items,",
        "one block, one label. Whichever label the model picks, the other item is lost.",
        "",
        "That is why *every* model struggles here, including the one that was perfect on",
        "sample 1. When all models fail on the same blocks, the problem sits before the models.",
        "",
        "### The overall lesson",
        "",
        "| document | who is at fault | how you can tell |",
        "|---|---|---|",
        "| sample 1 | the model | one model scored 100% from the same input |",
        "| sample 6 | the reader | every model fails on the same fused blocks |",
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
