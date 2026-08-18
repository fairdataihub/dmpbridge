"""Build notebooks/analysis-error-sources.ipynb — why two models score differently.

Three steps: the puzzle, the mechanism, what it means. Kept deliberately short.

Not currently built to disk as of 2026-08-18; run it if you want the file back.

    python scripts/build/build_error_analysis_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/analysis-error-sources.ipynb")


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
        "# Where do the errors come from?",
        "",
        "Every model gets **the same blocks** from the PDF. So any difference between them",
        "is the model, not the reading.",
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
        "from dmpbridge.evaluation.evaluate import load_method, micro_prf1",
        "",
        "MODELS, EXTRACTOR, SAMPLES = ['llama3.1:8b', 'gemma4:e4b', 'llama3.3:70b'], 'pdfplumber', range(1, 11)",
        "EXAMPLE = 5                     # document used for the example in step 2",
        "",
        "pd.set_option('display.max_colwidth', 60)",
        "INK = '#111111'",
        "TINT = {'title': '#e3edfa', 'section.title': '#fce8df',",
        "        'section.description': '#ddf3ec', 'question.text': '#ece3fa',",
        "        'answer.text': '#eceef0'}",
        "",
        "",
        "def tint(v):",
        "    \"\"\"One colour per label, so a run of one colour is visible at a glance.\"\"\"",
        "    return f'background-color: {TINT[v]}; color: {INK}' if v in TINT else ''",
        "",
        "",
        "available = [m for m in MODELS",
        "             if all(P.structured_path(P.make_tag(m, EXTRACTOR), n).exists()",
        "                    for n in SAMPLES)]",
        "print('models ready:', ', '.join(available))",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — The puzzle",
        "",
        "Same PDFs, same blocks, very different scores.",
    ]),
    code("step1", [
        "def label_runs(blocks):",
        "    \"\"\"How many times the label changes as you read down the document.\"\"\"",
        "    labs = [b.get('label') for b in blocks]",
        "    return 1 + sum(1 for i in range(1, len(labs)) if labs[i] != labs[i - 1])",
        "",
        "",
        "rows = []",
        "for m in available:",
        "    tag = P.make_tag(m, EXTRACTOR)",
        "    blocks = changes = 0",
        "    for n in SAMPLES:",
        "        bl = json.loads(P.labeled_path(tag, n).read_text(encoding='utf-8'))",
        "        blocks += len(bl)",
        "        changes += label_runs(bl)",
        "    rows.append({'model': m, 'blocks given': blocks,",
        "                 'label changes': changes,",
        "                 'changes per block': changes / blocks,",
        "                 'f1': micro_prf1(load_method(tag, exclude=[])[1])['f1']})",
        "",
        "df = pd.DataFrame(rows).set_index('model')",
        "display(df.style",
        "        .background_gradient(cmap='Reds', subset=['changes per block'])",
        "        .background_gradient(cmap='Greens', subset=['f1'])",
        "        .format({'changes per block': '{:.2f}', 'f1': '{:.3f}'}))",
    ]),
    md("md-1b", [
        "Every model was handed the **same number of blocks**. The model that changes its",
        "label most often scores worst.",
        "",
        "Step 2 shows why that matters so much.",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — Why changing labels is expensive",
        "",
        "pdfplumber reads **line by line**, so one paragraph arrives as several blocks.",
        "",
        "The pipeline can put them back together — it joins neighbouring blocks that have",
        "**the same label**. So:",
        "",
        "```",
        "answer  answer  answer  answer      ->  joined into 1  ->  correct",
        "answer  question  answer  question  ->  stays as 4     ->  1 right, 3 wrong",
        "```",
        "",
        "A model that keeps the same label through a paragraph gets it rebuilt for free.",
        "A model that flips back and forth leaves it in pieces, and every extra piece counts",
        "as an error.",
        "",
        "Here is that happening on a real document. **The text is identical** — only the",
        "labels differ.",
    ]),
    code("step2", [
        "labeled = {m: json.loads(P.labeled_path(P.make_tag(m, EXTRACTOR), EXAMPLE)",
        "                         .read_text(encoding='utf-8')) for m in available}",
        "n_blocks = len(next(iter(labeled.values())))",
        "",
        "# Show the stretch where the models disagree most.",
        "if len(available) >= 2:",
        "    a, b = available[0], available[1]",
        "    diff = [i for i in range(n_blocks)",
        "            if labeled[a][i].get('label') != labeled[b][i].get('label')]",
        "    start = max(0, (diff[len(diff) // 2] if diff else 0) - 5)",
        "else:",
        "    start = 0",
        "window = range(start, min(start + 12, n_blocks))",
        "",
        "ex = pd.DataFrame([{**{m: labeled[m][i].get('label') for m in available},",
        "                    'text': labeled[available[0]][i]['text']} for i in window],",
        "                  index=[f'block {i}' for i in window])",
        "display(ex.style.map(tint, subset=available))",
    ]),
    md("md-2b", [
        "Look down each model's column. **A single colour running down several rows** means",
        "those blocks get joined into one item — one correct answer.",
        "",
        "**Alternating colours** mean the same sentences stay split apart, and all but one of",
        "the pieces is counted wrong.",
    ]),

    # ── 3 ───────────────────────────────────────────────────────────────
    md("md-3", [
        "## Step 3 — What this means",
        "",
        "- **The PDF reader is not the problem.** It gives every model the same blocks, and",
        "  loses no text. It just cuts a paragraph into more pieces than the annotation has.",
        "",
        "- **Staying consistent is the skill being tested.** The pipeline repairs the cutting",
        "  by itself, but only for a model that holds one label across a line break.",
        "",
        "- **So a weak model looks far worse here than it really is at understanding text.**",
        "  Most of its errors are correct text, correctly read, broken into pieces.",
        "",
        "That is also why the line-merging step that used to run before labelling helped the",
        "small model so much and the larger ones barely at all — it was doing this joining in",
        "advance, for a model that could not do it itself.",
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
