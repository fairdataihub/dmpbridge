"""Build notebooks/8-confidence-analysis.ipynb — is the confidence score useful?

Every labeled block carries a confidence the model reported about its own label.
Path A, block level: does that number mean anything, and could it help?

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
        "# Does the model's confidence score mean anything?",
        "",
        "Every block the model labels comes with a **confidence** — the model's own claim",
        "about how sure it is. Three questions, in order:",
        "",
        "1. What values does each model actually give?",
        "2. When it says \"95%\", is it right 95% of the time? (*calibration*)",
        "3. Are wrong labels less confident than right ones? (*can it flag errors?*)",
        "",
        "Path A, block level: each block's true label comes from the annotation, exactly as",
        "in the error-analysis notebook.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import json",
        "import numpy as np",
        "import pandas as pd",
        "from scipy.stats import rankdata",
        "from IPython.display import display",
        "",
        "from dmpbridge.core import paths as P",
        "from dmpbridge.evaluation.evaluate import (",
        "    containment, extract_gold, resolve_old_gt_path, tokenize,",
        ")",
        "",
        "MODELS, EXTRACTOR, SAMPLES = ['llama3.1:8b', 'gemma4:e4b', 'llama3.3:70b'], 'pdfplumber', range(1, 11)",
        "pd.set_option('display.max_colwidth', 70)",
        "",
        "",
        "def block_truth(n):",
        "    \"\"\"True label per block, from the annotation. Tiny and item-spanning",
        "    blocks get None and are left out — same rules as the error notebook.\"\"\"",
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
        "    return out",
        "",
        "",
        "# One row per judgeable block: model, confidence, correct or not.",
        "rows = []",
        "for n in SAMPLES:",
        "    truth = block_truth(n)",
        "    for m in MODELS:",
        "        bl = json.loads(P.labeled_path(P.make_tag(m, EXTRACTOR), n)",
        "                        .read_text(encoding='utf-8'))",
        "        for t, b in zip(truth, bl):",
        "            if t is None:",
        "                continue",
        "            rows.append({'model': m, 'sample': n,",
        "                         'confidence': float(b.get('confidence', 1.0)),",
        "                         'correct': b.get('label') == t})",
        "df = pd.DataFrame(rows)",
        "print(f'{len(df) // len(MODELS)} judgeable blocks per model, {len(MODELS)} models')",
    ]),

    # ── 1 ───────────────────────────────────────────────────────────────
    md("md-1", [
        "## Step 1 — What values does each model give?",
    ]),
    code("step1", [
        "summary = df.groupby('model')['confidence'].agg(",
        "    ['mean', 'min', 'max', 'nunique']).reindex(MODELS)",
        "summary.columns = ['mean', 'lowest', 'highest', 'distinct values']",
        "summary['share at exactly 1.0'] = df.groupby('model')['confidence'] \\",
        "    .apply(lambda s: (s == 1.0).mean()).reindex(MODELS)",
        "display(summary.style.format({'mean': '{:.3f}', 'lowest': '{:.2f}',",
        "                              'highest': '{:.2f}',",
        "                              'share at exactly 1.0': '{:.0%}'}))",
    ]),
    md("md-1b", [
        "All three models do vary their confidence — none simply writes 1.0 everywhere.",
        "So the number is at least *trying* to say something. The next two steps test",
        "whether it succeeds.",
    ]),

    # ── 2 ───────────────────────────────────────────────────────────────
    md("md-2", [
        "## Step 2 — Calibration: when it says 90%, is it right 90% of the time?",
        "",
        "Blocks are grouped by the confidence the model claimed. In each group, the",
        "**accuracy should match the claim** — a model that says \"90%\" on a hundred blocks",
        "should be right on about ninety of them.",
    ]),
    code("step2", [
        "edges = [0, 0.7, 0.8, 0.9, 0.95, 0.999, 1.01]",
        "names = ['< 0.7', '0.7–0.8', '0.8–0.9', '0.9–0.95', '0.95–1.0', 'exactly 1.0']",
        "",
        "for m in MODELS:",
        "    d = df[df['model'] == m].copy()",
        "    d['claimed'] = pd.cut(d['confidence'], bins=edges, labels=names, right=False)",
        "    t = (d.groupby('claimed', observed=True)",
        "         .agg(blocks=('correct', 'size'), actually_right=('correct', 'mean'),",
        "              claimed_mean=('confidence', 'mean')))",
        "    t = t.rename(columns={'claimed_mean': 'claims on average',",
        "                          'actually_right': 'actually right'})",
        "    t['gap'] = t['actually right'] - t['claims on average']",
        "    display(t.style.set_caption(m)",
        "            .set_table_styles([{'selector': 'caption',",
        "                                'props': [('caption-side', 'top'),",
        "                                          ('font-weight', '700'),",
        "                                          ('font-size', '110%'),",
        "                                          ('text-align', 'left'),",
        "                                          ('color', '#111')]}])",
        "            .background_gradient(cmap='RdYlGn', subset=['gap'],",
        "                                 vmin=-0.5, vmax=0.5)",
        "            .format({'claims on average': '{:.2f}', 'actually right': '{:.0%}',",
        "                     'gap': '{:+.0%}'}))",
        "    print()",
    ]),
    md("md-2b", [
        "**Read the `gap` column** — actual accuracy minus claimed confidence. Red means",
        "over-confident: the model claims more certainty than it earns. A model can be",
        "roughly honest *on average* and still tell you nothing block by block — that is",
        "what step 3 tests.",
    ]),

    # ── 3 ───────────────────────────────────────────────────────────────
    md("md-3", [
        "## Step 3 — Can confidence flag the errors?",
        "",
        "The practical question. If wrong labels came with lower confidence, we could route",
        "low-confidence blocks to a human. That only works if the wrong ones actually *are*",
        "less confident.",
        "",
        "`AUC` below is the probability that a randomly chosen correct block has higher",
        "confidence than a randomly chosen wrong one: **0.5 = useless coin flip, 1.0 =",
        "perfect separation.**",
    ]),
    code("step3", [
        "rows = []",
        "for m in MODELS:",
        "    d = df[df['model'] == m]",
        "    ok, bad = d[d['correct']]['confidence'], d[~d['correct']]['confidence']",
        "    allv = np.concatenate([ok, bad])",
        "    ranks = rankdata(allv)",
        "    auc = (ranks[:len(ok)].sum() - len(ok) * (len(ok) + 1) / 2) \\",
        "          / (len(ok) * len(bad))",
        "    rows.append({'model': m,",
        "                 'mean conf when RIGHT': ok.mean(),",
        "                 'mean conf when WRONG': bad.mean(),",
        "                 'AUC': auc})",
        "sep = pd.DataFrame(rows).set_index('model')",
        "display(sep.style.background_gradient(cmap='RdYlGn', subset=['AUC'],",
        "                                      vmin=0.5, vmax=1.0)",
        "        .format('{:.3f}'))",
    ]),
    md("md-3b", [
        "And the what-if: suppose every block below a confidence threshold were sent to a",
        "human for review. How many errors would that catch, and how much correct work",
        "would be flagged along with them?",
    ]),
    code("step3b", [
        "rows = []",
        "for m in MODELS:",
        "    d = df[df['model'] == m]",
        "    ok, bad = d[d['correct']]['confidence'], d[~d['correct']]['confidence']",
        "    for thr in (0.85, 0.9, 0.95):",
        "        rows.append({'model': m, 'flag below': thr,",
        "                     'errors caught': (bad < thr).mean(),",
        "                     'correct blocks flagged too': (ok < thr).mean(),",
        "                     'blocks a human must review': (d['confidence'] < thr).mean()})",
        "what_if = pd.DataFrame(rows).set_index(['model', 'flag below'])",
        "display(what_if.style",
        "        .background_gradient(cmap='Greens', subset=['errors caught'])",
        "        .background_gradient(cmap='Reds',",
        "                             subset=['correct blocks flagged too'])",
        "        .format('{:.0%}'))",
    ]),

    # ── verdict ─────────────────────────────────────────────────────────
    md("verdict", [
        "## Verdict — does confidence help?",
        "",
        "**Only for gemma4:e4b.** The AUC row settles it:",
        "",
        "- **llama3.1:8b — no (AUC ≈ 0.54).** Its wrong labels are just as confident as its",
        "  right ones — on average slightly *more*. The number carries no signal.",
        "- **llama3.3:70b — no (AUC ≈ 0.53).** It spreads its confidence over a wide range,",
        "  and is roughly honest on average, but the spread does not line up with which",
        "  blocks are wrong. Calibrated overall, uninformative per block.",
        "- **gemma4:e4b — yes (AUC ≈ 0.80).** Its errors really are less confident. Flagging",
        "  its blocks below 0.95 would catch roughly three quarters of its errors while",
        "  flagging about a third of its output for review.",
        "",
        "**Suggestion.** Use confidence for one thing only: **routing gemma4's low-confidence",
        "blocks to human review** — not for auto-dropping blocks (a dropped block becomes a",
        "missed item, which costs the same as an error), and not for the other two models,",
        "where the score is noise. And never compare confidence across models: each model's",
        "scale means something different, so a shared threshold is meaningless.",
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
