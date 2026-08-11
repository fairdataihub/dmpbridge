"""Build notebooks/7-sample1-error-analysis.ipynb.

One document, three models, five labels. Answers the question: is an error the
extractor's fault or the model's? Kept short and concrete.

    python scripts/build/build_sample_analysis_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/7-sample1-error-analysis.ipynb")


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
        "# Sample 1 — where does each error come from?",
        "",
        "One document, three models, five labels.",
        "",
        "There are only two places an error can start:",
        "",
        "1. **the extractor** — pdfplumber reads line by line, so it can cut one item into",
        "   several blocks",
        "2. **the model** — it can give a block the wrong label",
        "",
        "This notebook separates them.",
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
        "    LABELS, _confusion_from_match, _match_structured, containment,",
        "    extract_gold, micro_prf1, resolve_old_gt_path, tokenize,",
        ")",
        "",
        "SAMPLE, EXTRACTOR = 1, 'pdfplumber'",
        "MODELS = ['llama3.1:8b', 'gemma4:e4b', 'llama3.3:70b']",
        "",
        "pd.set_option('display.max_colwidth', 72)",
        "INK, WARN, GOOD = '#111111', '#fdecea', '#e6f6ee'",
        "",
        "gold = extract_gold(resolve_old_gt_path(SAMPLE))",
        "blocks = json.loads((P.EXTRACTED_DIR / EXTRACTOR / f'sample{SAMPLE}.json')",
        "                    .read_text(encoding='utf-8'))",
        "available = [m for m in MODELS",
        "             if P.structured_path(P.make_tag(m, EXTRACTOR), SAMPLE).exists()]",
        "",
        "print(f'sample{SAMPLE}: {len(gold)} items in the annotation, "
        "{len(blocks)} blocks from pdfplumber')",
        "print('models:', ', '.join(available))",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — What the extractor did",
        "",
        "The annotation lists **items** — one answer is one item, however many lines it takes",
        "on the page. pdfplumber gives back **blocks**, one per line.",
        "",
        "So: which labels got cut up?",
    ]),
    code("step1", [
        "# Match every block to the annotation item it belongs to.",
        "owner = {}",
        "for i, b in enumerate(blocks):",
        "    bt = tokenize(b['text'])",
        "    best, bi = 0.0, None",
        "    for gi, (gt, _) in enumerate(gold):",
        "        c = containment(bt, tokenize(gt))",
        "        if c > best:",
        "            best, bi = c, gi",
        "    owner[i] = bi if best >= 0.75 else None",
        "",
        "rows = []",
        "for lab in LABELS:",
        "    idx = [gi for gi, (_, l) in enumerate(gold) if l == lab]",
        "    if not idx:",
        "        continue",
        "    counts = [sum(1 for v in owner.values() if v == gi) for gi in idx]",
        "    rows.append({'label': lab, 'items': len(idx), 'blocks': sum(counts),",
        "                 'items that got split': sum(1 for c in counts if c > 1),",
        "                 'worst split': max(counts)})",
        "",
        "step1 = pd.DataFrame(rows).set_index('label')",
        "display(step1.style.background_gradient(cmap='Oranges',",
        "                                        subset=['items that got split']))",
    ]),
    md("md-1b", [
        "**Only answers get cut up.** Titles, headings and questions are one line each, so they",
        "arrive whole. Answers are paragraphs, so they arrive in pieces.",
        "",
        "That already narrows things down: **any error on a title, heading or question cannot",
        "be the extractor's fault** — those blocks were handed over intact.",
        "",
        "It also means the extractor's damage, if any, is confined to `answer.text`.",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — What each model did",
        "",
        "For every item in the annotation: did the model get it, give it the wrong label, or",
        "miss it? And how many extra items did it produce that should not exist?",
    ]),
    code("step2", [
        "def per_label(model):",
        "    tag = P.make_tag(model, EXTRACTOR)",
        "    rec, extra = _match_structured(P.structured_path(tag, SAMPLE), gold)",
        "    m = micro_prf1(_confusion_from_match(rec, extra))",
        "    rows = []",
        "    for lab in LABELS:",
        "        items = [r for r in rec if r['gold_label'] == lab]",
        "        if not items:",
        "            continue",
        "        rows.append({",
        "            'label': lab,",
        "            'items': len(items),",
        "            'correct': sum(1 for r in items if r['pred_label'] == lab),",
        "            'wrong label': sum(1 for r in items",
        "                               if r['pred_label'] and r['pred_label'] != lab),",
        "            'missing': sum(1 for r in items if r['pred_label'] is None),",
        "            'extra produced': sum(1 for _, l in extra if l == lab),",
        "        })",
        "    return pd.DataFrame(rows).set_index('label'), m",
        "",
        "",
        "for mdl in available:",
        "    tbl, m = per_label(mdl)",
        "    display(tbl.style",
        "            .set_caption(f'{mdl}  —  f1 {m[\"f1\"]:.3f}')",
        "            .set_table_styles([{'selector': 'caption',",
        "                                'props': [('caption-side', 'top'),",
        "                                          ('font-size', '112%'),",
        "                                          ('font-weight', '700'),",
        "                                          ('text-align', 'left'),",
        "                                          ('padding-bottom', '6px'),",
        "                                          ('color', INK)]}])",
        "            .background_gradient(cmap='Greens', subset=['correct'])",
        "            .background_gradient(cmap='Reds',",
        "                                 subset=['wrong label', 'missing', 'extra produced']))",
        "    print()",
    ]),

    # ── 3 ───────────────────────────────────────────────────────────────
    md("md-3", [
        "## Step 3 — So whose fault is it?",
        "",
        "Look at the `answer.text` row — the only label the extractor cut up.",
        "",
        "If the splitting were the problem, **every** model would lose points there. Compare",
        "the three tables above and see whether that is what happened.",
    ]),
    code("step3", [
        "rows = []",
        "for mdl in available:",
        "    tbl, m = per_label(mdl)",
        "    errs = tbl[['wrong label', 'missing', 'extra produced']].sum(axis=1)",
        "    rows.append({'model': mdl, 'f1': m['f1'],",
        "                 'errors on answer.text  (the split label)': int(errs.get('answer.text', 0)),",
        "                 'errors on everything else': int(errs.sum() - errs.get('answer.text', 0))})",
        "",
        "verdict = pd.DataFrame(rows).set_index('model')",
        "display(verdict.style.format({'f1': '{:.3f}'})",
        "        .background_gradient(cmap='Greens', subset=['f1']))",
    ]),
    md("md-3b", [
        "**The splitting is not what hurts.** The extractor cut this document's answers into",
        "many pieces and handed every model the same pieces — yet the strongest models lose",
        "almost nothing on `answer.text`.",
        "",
        "That is because the pipeline rejoins neighbouring blocks that carry the **same label**.",
        "A model that stays consistent through a paragraph gets it rebuilt for free. Only a",
        "model that changes label part-way through leaves it in pieces.",
        "",
        "So the fragmentation is an *opportunity* for an error, not a cause of one.",
    ]),

    # ── 4 ───────────────────────────────────────────────────────────────
    md("md-4", [
        "## Step 4 — The errors that are left",
        "",
        "Everything not explained above is the model choosing the wrong label for text it was",
        "handed intact. Here is each one, with what it should have been.",
    ]),
    code("step4", [
        "for mdl in available:",
        "    tag = P.make_tag(mdl, EXTRACTOR)",
        "    rec, extra = _match_structured(P.structured_path(tag, SAMPLE), gold)",
        "    bad = [{'should be': r['gold_label'], 'model said': r['pred_label'],",
        "            'text': r['pred_text']}",
        "           for r in rec if r['pred_label'] and r['pred_label'] != r['gold_label']]",
        "    print(f'{mdl} — {len(bad)} wrong label(s)')",
        "    if bad:",
        "        display(pd.DataFrame(bad))",
        "    print()",
    ]),
    md("md-4b", [
        "---",
        "",
        "**The conclusion for this document.** The extractor did its job — it read every word",
        "and lost nothing; it only cut the answers into lines. Where a model labels those lines",
        "consistently, the pipeline puts them back and the score is unaffected.",
        "",
        "What separates the models here is a single decision: whether a short, bold, lettered",
        "line such as `A. Types and amount of scientific data…` is a **question** or a",
        "**section heading**. That is a judgement about the document's structure, and it is",
        "the model's to make.",
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
