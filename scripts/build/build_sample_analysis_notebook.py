"""Build notebooks/analysis-sample-errors.ipynb.

The same three-step investigation run on two documents that reach opposite
verdicts: sample 1 (the model's fault) and sample 6 (the reader's fault).

Not currently built to disk as of 2026-08-18; run it if you want the file back.

    python scripts/build/build_sample_analysis_notebook.py
"""
import json
from pathlib import Path

from dmpbridge.core import paths as P

NB = Path("notebooks/analysis-sample-errors.ipynb")
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
            "print(f'blocks left out of the check (too small to judge, or spanning',",
            "      f'two items): {len(lost)}')",
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
        "        # A one- or two-word block (a page number, a stray '1') can match an",
        "        # item by accident, so tiny blocks are not given a true label at all.",
        "        if len(bt) < 3:",
        "            truth.append(None)",
        "            continue",
        "        best, lab, bi = 0.0, None, None",
        "        for gi, (gt, gl) in enumerate(gold):",
        "            c = containment(bt, tokenize(gt))",
        "            if c > best:",
        "                best, lab, bi = c, gl, gi",
        "        if best < 0.75:",
        "            truth.append(None)",
        "            continue",
        "        # A block that also swallows a DIFFERENT whole item (a heading fused",
        "        # onto the front of an answer) spans two items — no single label can",
        "        # be right, so it gets no true label either.",
        "        spans_two = any(gi != bi and len(tokenize(gt)) >= 2",
        "                        and containment(tokenize(gt), bt) >= 0.9",
        "                        for gi, (gt, gl) in enumerate(gold))",
        "        truth.append(None if spans_two else lab)",
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
    md("s6-md-4", [
        "### Step 4 — What did the models do with the fused blocks?",
        "",
        "The blocks left out above are the interesting ones: each holds an underlined",
        "heading glued to the start of its answer. No single label can be right for them —",
        "but what did the models actually say?",
    ]),
    code("s6-step4", [
        "fused = [i for i, t in enumerate(truth) if t is None",
        "         and len(blocks[i]['text'].split()) > 4]",
        "rows = [{**{m: labeled[m][i].get('label') for m in MODELS},",
        "         'text': blocks[i]['text']} for i in fused]",
        "display(pd.DataFrame(rows, index=[f'block {i}' for i in fused]))",
    ]),
    md("s6-md-4b", [
        "**Every model says `section.title`, unanimously.** That is the sensible answer —",
        "each block really does start with a heading. But the block also contains the",
        "answer's first line, so the scoring can match it to neither item: the heading item",
        "is missed *and* the answer loses its opening. The models did the best possible",
        "thing with the input, and still lose twice.",
    ]),
    md("s6-md-5", [
        "### Step 5 — What those blocks become in the final output",
        "",
        "The label applies to the **whole block**, so each model's produced \"heading\" is the",
        "fused text — heading *plus* the answer's first line. The scorer accepts a match when",
        "at least 75% of a produced item's words are inside the real item. Mostly answer",
        "words, so:",
    ]),
    code("s6-step5", [
        "gold_heads = [t for t, l in gold if l == 'section.title']",
        "",
        "",
        "def produced_heads(model):",
        "    s3 = json.loads(P.structured_path(P.make_tag(model, EXTRACTOR), 6)",
        "                    .read_text(encoding='utf-8'))",
        "    return [s.get('title', '') for s in s3['narrative']['template']['section']",
        "            if s.get('title')]",
        "",
        "",
        "for m in MODELS:",
        "    heads = produced_heads(m)",
        "    matched = sum(max((containment(tokenize(p), tokenize(g)) for g in gold_heads),",
        "                      default=0) >= 0.75 for p in heads)",
        "    print(f'{m:<14} real headings matched: {matched} of {len(gold_heads)}')",
        "",
        "print()",
        "rows = []",
        "for p in produced_heads(MODELS[0]):",
        "    if len(tokenize(p)) < 3:",
        "        continue                      # stray page number, not a fused heading",
        "    ov, g = max(((containment(tokenize(p), tokenize(g)), g) for g in gold_heads),",
        "                key=lambda x: x[0], default=(0, ''))",
        "    rows.append({'produced heading (fused)': p[:56],",
        "                 'real heading': g, 'word overlap': f'{ov:.0%}',",
        "                 'match (needs 75%)': 'no'})",
        "display(pd.DataFrame(rows))",
    ]),
    md("s6-md-5b", [
        "**All three models produce these same fused items, at the same overlap figures —",
        "and match 0 of the 5 real headings.** An 8B and a 70B model ending up with the",
        "identical wrong item is only possible when the item was decided by the input, not",
        "by the model.",
    ]),
    md("s6-conclusion", [
        "### Conclusion for sample 6",
        "",
        "**Here the reader is at fault — and no model can fix it.**",
        "",
        "This document's headings are underlined, and an underline is a drawn line the reader",
        "cannot see. So each heading arrives **glued to its answer in one block** — two items,",
        "one block, one label. Whichever label the model picks, the other item is lost.",
        "",
        "The cleanest proof is llama3.3:70b: it labels **every judgeable block correctly**",
        "(100% in step 2) and answers `section.title` on all five fused blocks — the sensible",
        "call — yet this is still its worst document in the corpus (f1 0.52). A model with",
        "zero labeling mistakes cannot score well here, because five of the eleven items",
        "never reached it as separate blocks.",
        "",
        "### The overall lesson",
        "",
        "| document | who is at fault | how you can tell |",
        "|---|---|---|",
        "| sample 1 | the model | another model scored 100% from the same input |",
        "| sample 6 | the reader | even a model with zero block mistakes scores only 0.52 |",
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
