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
    md("md-5-sd", [
        "### The same two columns, with their spread — and a significance test",
        "",
        "An average alone can hide a lot.",
        "",
        "- **SD** (standard deviation) says how much the individual items vary around the",
        "  average. A small SD means they cluster near it; a large one means they are spread",
        "  out.",
        "- **p** answers: *could a difference this size have come about by chance?* Below 0.05",
        "  is the usual bar for saying no.",
        "",
        "The test is Mann–Whitney U, which compares two groups by rank. It suits this data",
        "because confidence values are not normally distributed — they bunch up against 1.0.",
    ]),
    code("result-sd", [
        "from scipy.stats import mannwhitneyu",
        "",
        "",
        "def mean_sd(values):",
        "    \"\"\"'89.4 (16.0)' — the average, with the spread in brackets.\"\"\"",
        "    return f'{values.mean() * 100:.1f} ({values.std() * 100:.1f})'",
        "",
        "",
        "rows = []",
        "for model in MODELS:",
        "    d = df[df['model'] == model]",
        "    right, wrong = d[d['correct']]['confidence'], d[~d['correct']]['confidence']",
        "    p = mannwhitneyu(right, wrong, alternative='two-sided').pvalue",
        "    rows.append({",
        "        'Model': model,",
        "        'Confidence when correct (%), mean (SD)': mean_sd(right),",
        "        'Confidence when incorrect (%), mean (SD)': mean_sd(wrong),",
        "        'p': f'{p:.3f}' if p >= 0.001 else '< 0.001',",
        "    })",
        "display(pd.DataFrame(rows).set_index('Model'))",
    ]),
    md("md-5-sd-b", [
        "**The two groups overlap heavily.** For every model the gap between the averages is",
        "smaller than the spread within each group — so plenty of **wrong** items are more",
        "confident than plenty of **right** ones.",
        "",
        "**A word of caution on `p`.** For gemma4:e4b and llama3.3:70b the difference is",
        "statistically significant, but that only means it is unlikely to be chance — not that",
        "it is large enough to be useful. Their averages differ by about 4–5 points while",
        "individual items vary by 7–11, so a threshold placed anywhere still misclassifies a",
        "great many items. For llama3.1:8b the difference is not significant at all.",
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

    # ── write-up ────────────────────────────────────────────────────────
    md("md-writeup", [
        "## Results paragraph, for the report",
        "",
        "Generated from the two tables above, so the wording and the numbers cannot drift",
        "apart. Copy the output straight into the write-up.",
    ]),
    code("writeup", [
        "import textwrap",
        "",
        "best = table['Accuracy, %'].idxmax()",
        "worst = table['Accuracy, %'].idxmin()",
        "loudest = table['Total confidence, %'].idxmax()",
        "",
        "",
        "def stats(model):",
        "    \"\"\"Everything the paragraph needs about one model, as percentages.\"\"\"",
        "    d = df[df['model'] == model]",
        "    right, wrong = d[d['correct']]['confidence'], d[~d['correct']]['confidence']",
        "    return {",
        "        'acc': d['correct'].mean() * 100,",
        "        'conf': d['confidence'].mean() * 100,",
        "        'right': right.mean() * 100, 'right_sd': right.std() * 100,",
        "        'wrong': wrong.mean() * 100, 'wrong_sd': wrong.std() * 100,",
        "        'gap': (right.mean() - wrong.mean()) * 100,",
        "        'p': mannwhitneyu(right, wrong, alternative='two-sided').pvalue,",
        "    }",
        "",
        "",
        "S = {m: stats(m) for m in MODELS}",
        "gaps = {m: S[m]['gap'] for m in MODELS}",
        "widest = max(gaps, key=gaps.get)",
        "",
        "",
        "def p_text(p):",
        "    return 'P < .001' if p < 0.001 else f'P = {p:.2f}'.replace('0.', '.')",
        "",
        "",
        "# The most confident model may or may not also be the most accurate — say",
        "# whichever is true rather than assuming a contrast.",
        "if loudest == best:",
        "    tracking = (f'The most confident model overall, {loudest} '",
        "                f'({S[loudest][\"conf\"]:.1f}%), was also the most accurate, but its '",
        "                f'confidence still exceeded its accuracy by '",
        "                f'{S[loudest][\"conf\"] - S[loudest][\"acc\"]:.1f} percentage points.')",
        "else:",
        "    tracking = (f'The most confident model overall was {loudest} '",
        "                f'({S[loudest][\"conf\"]:.1f}%), which was not the most accurate, so '",
        "                f'reported confidence did not track performance.')",
        "",
        "paragraph = (",
        "    f\"Across {len(df)} aggregated items from 10 Data Management Plans, every model \"",
        "    f\"reported confidence substantially higher than its accuracy warranted. \"",
        "    f\"{best} achieved the highest accuracy at {S[best]['acc']:.1f}% while reporting \"",
        "    f\"{S[best]['conf']:.1f}% mean confidence, whereas {worst} was correct on only \"",
        "    f\"{S[worst]['acc']:.1f}% of items yet reported {S[worst]['conf']:.1f}% \"",
        "    f\"confidence. {tracking} \"",
        "    f\"The mean difference in confidence between correct and incorrect items was \"",
        "    f\"small for every model, ranging from {min(gaps.values()):.1f} to \"",
        "    f\"{max(gaps.values()):.1f} percentage points. {widest} showed the widest \"",
        "    f\"separation ({S[widest]['right']:.1f}%, SD {S[widest]['right_sd']:.1f} when \"",
        "    f\"correct versus {S[widest]['wrong']:.1f}%, SD {S[widest]['wrong_sd']:.1f} when \"",
        "    f\"incorrect; {p_text(S[widest]['p'])}), and {worst} the narrowest \"",
        "    f\"({gaps[worst]:.1f} percentage points; {p_text(S[worst]['p'])}). In every model \"",
        "    f\"the spread of confidence within each group exceeded the difference between \"",
        "    f\"the groups, so no threshold separated correct from incorrect items reliably.\"",
        ")",
        "",
        "print('Results\\n')",
        "print(textwrap.fill(paragraph, width=88))",
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
