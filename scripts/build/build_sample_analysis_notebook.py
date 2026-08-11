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
        "INK, WRONG = '#111111', '#fdecea'      # wrong rows tinted, right rows left plain",
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
        "def side_by_side(model):",
        "    \"\"\"Every annotation item next to the label the model gave that same text.",
        "",
        "    Matching is by shared words, not by row number — the model often produces",
        "    a different number of items, so the two lists drift out of step and",
        "    comparing row 12 with row 12 would compare unrelated things.",
        "    \"\"\"",
        "    tag = P.make_tag(model, EXTRACTOR)",
        "    rec, _ = _match_structured(P.structured_path(tag, SAMPLE), gold)",
        "    return pd.DataFrame([{",
        "        'annotation says': r['gold_label'],",
        "        'model said': r['pred_label'] if r['pred_label'] else 'not produced',",
        "        'ok': '' if r['pred_label'] == r['gold_label'] else 'X',",
        "        'text': r['gold_text'],",
        "    } for r in rec])",
        "",
        "",
        "def flag(row):",
        "    \"\"\"Tint the whole row when the model got it wrong.\"\"\"",
        "    bad = row['ok'] == 'X'",
        "    return [f'background-color: {WRONG}; color: {INK}' if bad else '' for _ in row]",
        "",
        "",
        "def show(model):",
        "    \"\"\"Score, then every item, then the extra items.\"\"\"",
        "    df, m = mistakes(model)",
        "    print(f'{len(gold)} items in the annotation')",
        "    print(f'  {m[\"tp\"]:>2} correct')",
        "    print(f'  {len(df):>2} wrong')",
        "    print(f'  f1 = {m[\"f1\"]:.3f}')",
        "",
        "    print('\\nEvery item, annotation against model:')",
        "    display(side_by_side(model).style.apply(flag, axis=1))",
        "",
        "    tag = P.make_tag(model, EXTRACTOR)",
        "    _, extra = _match_structured(P.structured_path(tag, SAMPLE), gold)",
        "    if extra:",
        "        print(f'Plus {len(extra)} item(s) the model produced that the annotation '",
        "              f'does not contain:')",
        "        display(pd.DataFrame([{'model said': l, 'text': t} for t, l in extra]))",
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
    "**How the two are lined up**",
    "",
    "Each row pairs one annotation item with the label the model gave *that same text*.",
    "The pairing is done by shared words, not by position — the model usually produces a",
    "different number of items, so the two lists drift out of step and comparing row 12",
    "against row 12 would compare unrelated things.",
    "",
    "| column | meaning |",
    "|---|---|",
    "| `annotation says` | the correct label |",
    "| `model said` | what the model called that text, or `not produced` if it never appeared |",
    "| `ok` | `X` marks a row where the two disagree |",
    "",
    "Items listed separately at the end are ones the model produced that have no",
    "counterpart in the annotation at all.",
]))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells, sections for {', '.join(AVAILABLE)}")
