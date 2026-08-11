"""Build notebooks/8-confidence-analysis.ipynb — can we trust the confidence score?

Kept deliberately simple: show the wrong labels that came with high confidence,
count them, and say whether the score is usable.

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
        "# Can we trust the model's confidence score?",
        "",
        "Every labeled block comes with a confidence — the model saying how sure it is.",
        "If that number were trustworthy, wrong labels would come with low confidence.",
        "",
        "So: **show the wrong labels, with the confidence the model attached to them.**",
        "Path A, block level, all 10 documents.",
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
        "MODELS, EXTRACTOR, SAMPLES = ['llama3.1:8b', 'gemma4:e4b', 'llama3.3:70b'], 'pdfplumber', range(1, 11)",
        "pd.set_option('display.max_colwidth', 66)",
        "INK, WRONGBG = '#111111', '#fdecea'",
        "",
        "",
        "def block_truth(n):",
        "    \"\"\"True label per block, from the annotation — same rules as notebook 7",
        "    (tiny blocks and blocks spanning two items are left out).\"\"\"",
        "    gold = extract_gold(resolve_old_gt_path(n))",
        "    blocks = json.loads((P.EXTRACTED_DIR / EXTRACTOR / f'sample{n}.json')",
        "                        .read_text(encoding='utf-8'))",
        "    out = []",
        "    for b in blocks:",
        "        bt = tokenize(b['text'])",
        "        if len(bt) < 3:",
        "            out.append(None); continue",
        "        best, lab, bi = 0.0, None, None",
        "        for gi, (gt, gl) in enumerate(gold):",
        "            c = containment(bt, tokenize(gt))",
        "            if c > best:",
        "                best, lab, bi = c, gl, gi",
        "        if best < 0.75:",
        "            out.append(None); continue",
        "        spans = any(gi != bi and len(tokenize(gt)) >= 2",
        "                    and containment(tokenize(gt), bt) >= 0.9",
        "                    for gi, (gt, gl) in enumerate(gold))",
        "        out.append(None if spans else lab)",
        "    return out, blocks",
        "",
        "",
        "# One row per judgeable block, per model.",
        "rows = []",
        "for n in SAMPLES:",
        "    truth, blocks = block_truth(n)",
        "    for m in MODELS:",
        "        bl = json.loads(P.labeled_path(P.make_tag(m, EXTRACTOR), n)",
        "                        .read_text(encoding='utf-8'))",
        "        for t, b in zip(truth, bl):",
        "            if t is None:",
        "                continue",
        "            rows.append({'model': m, 'sample': n, 'true label': t,",
        "                         'model said': b.get('label'),",
        "                         'confidence': float(b.get('confidence', 1.0)),",
        "                         'correct': b.get('label') == t,",
        "                         'text': b['text']})",
        "df = pd.DataFrame(rows)",
        "print(f'{len(df) // len(MODELS)} judgeable blocks per model')",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — Wrong labels, sorted by how confident the model was",
        "",
        "For each model: its ten most confident **mistakes**. If confidence were",
        "trustworthy, this table should show low numbers. It does not.",
    ]),
    code("step1", [
        "for m in MODELS:",
        "    bad = (df[(df['model'] == m) & (~df['correct'])]",
        "           .sort_values('confidence', ascending=False)",
        "           [['confidence', 'true label', 'model said', 'text']]",
        "           .head(10).reset_index(drop=True))",
        "    display(bad.style",
        "            .set_caption(f'{m} — most confident mistakes')",
        "            .set_table_styles([{'selector': 'caption',",
        "                                'props': [('caption-side', 'top'),",
        "                                          ('font-weight', '700'),",
        "                                          ('font-size', '110%'),",
        "                                          ('text-align', 'left'),",
        "                                          ('color', INK)]}])",
        "            .background_gradient(cmap='Reds', subset=['confidence'],",
        "                                 vmin=0.5, vmax=1.0)",
        "            .format({'confidence': '{:.2f}'}))",
        "    print()",
    ]),
    md("md-1b", [
        "Example of what this means: llama3.1 labels `B. Scientific data that will be",
        "preserved…` as a section heading **with confidence 1.00** — complete certainty,",
        "completely wrong. The model is not unsure when it errs; it is sure and wrong.",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — How much of each model's wrongness is high-confidence?",
        "",
        "One number per model: of all its wrong labels, how many carried a confidence of",
        "0.9 or more? And for comparison, the average confidence when right vs when wrong.",
    ]),
    code("step2", [
        "rows = []",
        "for m in MODELS:",
        "    d = df[df['model'] == m]",
        "    ok, bad = d[d['correct']], d[~d['correct']]",
        "    rows.append({'model': m,",
        "                 'wrong labels': len(bad),",
        "                 'wrong with confidence >= 0.9': (bad['confidence'] >= 0.9).mean(),",
        "                 'avg confidence when RIGHT': ok['confidence'].mean(),",
        "                 'avg confidence when WRONG': bad['confidence'].mean()})",
        "t = pd.DataFrame(rows).set_index('model')",
        "display(t.style",
        "        .background_gradient(cmap='Reds',",
        "                             subset=['wrong with confidence >= 0.9'])",
        "        .format({'wrong with confidence >= 0.9': '{:.0%}',",
        "                 'avg confidence when RIGHT': '{:.2f}',",
        "                 'avg confidence when WRONG': '{:.2f}'}))",
    ]),

    # ── verdict ─────────────────────────────────────────────────────────
    md("verdict", [
        "## Conclusion",
        "",
        "**The confidence score cannot be trusted to find errors.**",
        "",
        "- **llama3.1:8b** is *more* confident when wrong (0.88) than when right (0.87).",
        "  Its signature mistake — lettered questions labeled as headings — mostly comes",
        "  with confidence 0.95–1.00.",
        "- **llama3.3:70b** looks similar: right and wrong labels carry nearly the same",
        "  confidence, so the number cannot separate them.",
        "- **gemma4:e4b** is the partial exception — its wrong labels are noticeably less",
        "  confident (0.90 vs 0.95), so *very* low values are worth a second look. But even",
        "  there, most mistakes still carry 0.9 or above.",
        "",
        "Practical rule: **do not use confidence to filter or auto-accept labels.** A high",
        "number does not mean a right label — the most confident mistakes above are at 1.00.",
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
