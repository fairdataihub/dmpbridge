"""Build notebooks/7-sample1-error-analysis.ipynb.

One document. One section per model, listing every mistake it made and what the
text should have been. Nothing else.

    python scripts/build/build_sample_analysis_notebook.py
"""
import json
from pathlib import Path

from dmpbridge.core import paths as P

NB = Path("notebooks/7-sample1-error-analysis.ipynb")
SAMPLE, EXTRACTOR = 1, "pdfplumber"
MODELS = ["llama3.1:8b", "gemma4:e4b", "llama3.3:70b"]

# Emit a section only for models that have actually been run.
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
        f"# Sample {SAMPLE} — every mistake, model by model",
        "",
        f"The annotation for sample {SAMPLE} lists a fixed set of items. For each model below:",
        "what it got right, and every single thing it got wrong.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import pandas as pd",
        "from IPython.display import display",
        "",
        "from dmpbridge.core import paths as P",
        "from dmpbridge.evaluation.evaluate import (",
        "    _confusion_from_match, _match_structured, extract_gold, micro_prf1,",
        "    resolve_old_gt_path,",
        ")",
        "",
        f"SAMPLE, EXTRACTOR = {SAMPLE}, {EXTRACTOR!r}",
        "pd.set_option('display.max_colwidth', 90)",
        "gold = extract_gold(resolve_old_gt_path(SAMPLE))",
        "",
        "",
        "def mistakes(model):",
        "    \"\"\"Every wrong item this model produced, and the score for the document.\"\"\"",
        "    tag = P.make_tag(model, EXTRACTOR)",
        "    rec, extra = _match_structured(P.structured_path(tag, SAMPLE), gold)",
        "    m = micro_prf1(_confusion_from_match(rec, extra))",
        "    rows = [{'should be': r['gold_label'], 'model said': r['pred_label'],",
        "             'text': r['pred_text']}",
        "            for r in rec if r['pred_label'] and r['pred_label'] != r['gold_label']]",
        "    rows += [{'should be': 'nothing — extra item', 'model said': l, 'text': t}",
        "             for t, l in extra]",
        "    rows += [{'should be': r['gold_label'], 'model said': 'not produced',",
        "              'text': r['gold_text']}",
        "             for r in rec if r['pred_label'] is None]",
        "    return pd.DataFrame(rows), m",
        "",
        "",
        "def show(model):",
        "    \"\"\"Print the score, then list the mistakes.\"\"\"",
        "    df, m = mistakes(model)",
        "    print(f'{len(gold)} items in the annotation')",
        "    print(f'  {m[\"tp\"]:>2} correct')",
        "    print(f'  {len(df):>2} wrong')",
        "    print(f'  f1 = {m[\"f1\"]:.3f}')",
        "    if len(df):",
        "        print()",
        "        display(df)",
        "    else:",
        "        print('\\nNo mistakes.')",
        "",
        "",
        "print('sections below:', ', '.join("
        + repr(AVAILABLE) + "))",
    ]),
]

for i, model in enumerate(AVAILABLE, start=1):
    slug = model.replace(":", "-").replace(".", "")
    cells.append(md(f"md-{slug}", [f"## {model}"]))
    cells.append(code(f"code-{slug}", [f"show({model!r})"]))

cells.append(md("closing", [
    "---",
    "",
    "**How to read the `should be` column**",
    "",
    "| value | meaning |",
    "|---|---|",
    "| a label name | the annotation has this text as that label; the model called it something else |",
    "| `nothing — extra item` | the model produced an item the annotation does not have |",
    "| paired with `not produced` | the annotation has this item and the model never produced it |",
]))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells, sections for {', '.join(AVAILABLE)}")
