"""Build notebooks/analysis-gemma4-e4b-sample-by-sample.ipynb.

A plain-language walkthrough of gemma4:e4b's FINAL output (stage 4 — the
actual finished document the pipeline produces, after the automatic fix-up
rules run) against the answer key it's graded against (Path B only).

Each sample is shown as the real document structure — title, then each
section's heading, question and answer — in reading order, with a checkmark
or a plain-English flag next to anything that doesn't match the answer key.
No abstract statistics, no Path A, no separate "confusion matrix" section —
just the document you'd actually open, annotated.

    python scripts/build/build_gemma_sample_walkthrough.py
"""
import json
from pathlib import Path

NB = Path("notebooks/analysis-gemma4-e4b-sample-by-sample.ipynb")
MODEL = "gemma4:e4b"
TAG = "gemma4-e4b_pdfplumber_whole_doc"


def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = [
    md("title", [
        f"# {MODEL} — the actual final document, sample by sample",
        "",
        "This shows exactly what you'd see if you opened the finished output file for",
        "each of the 10 sample documents — title, then each section with its question",
        "and answer — with a checkmark or a flag next to anything that doesn't match",
        "the human-written answer key it's checked against.",
        "",
        "**\"Final\" and \"Path B\" here mean the same thing**: the pipeline's last stage,",
        "after a small automatic fix-up step runs (filling in a question that was left",
        "blank, using the section heading above it). That's the file this notebook",
        "reads — nothing here is the model's raw, unfixed output.",
    ]),

    md("md-legend", [
        "**Reading the flags below:**",
        "",
        "| Flag | Meaning |",
        "|---|---|",
        "| ✓ | Matches the answer key |",
        "| ⚠️ wrong label | The text is right, but it's labelled as the wrong kind of thing |",
        "| ⚠️ doesn't match the answer key | This text isn't in the answer key at all, or doesn't match closely enough |",
        "| ⚠️ starts lowercase | A simple warning sign for a real question: real questions start with a capital",
        "| | letter. One that doesn't is often a leftover sentence fragment, not a real question. |",
        "",
        "At the end of each sample, anything the answer key expects that isn't in the",
        "document at all is listed separately, under **missing**.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import json",
        "",
        "from dmpbridge.core import paths as P",
        "from dmpbridge.evaluation.evaluate import extract_gold, _match_structured",
        "from dmpbridge.evaluation.annotation_rules import resolve_new_gt_path, convert_tag_to_final",
        "",
        f"MODEL, TAG = {MODEL!r}, {TAG!r}",
        "SAMPLES = range(1, 11)",
        "",
        "if not P.final_path(TAG, 1).exists():",
        "    convert_tag_to_final(TAG)",
        "",
        "",
        "def sample_status(n):",
        "    \"\"\"(status_lookup, missing) for one sample's final JSON vs its answer key.",
        "",
        "    status_lookup maps (text, label) -> a short flag string, built from every",
        "    predicted item; missing lists answer-key items nothing in the document",
        "    matched. Path B only: dedup_question_title=False is Path B's own setting,",
        "    since it exists specifically to measure the fill-in step Path A skips.",
        "    \"\"\"",
        "    gold = extract_gold(resolve_new_gt_path(n), dedup_question_title=False)",
        "    records, no_gold = _match_structured(P.final_path(TAG, n), gold, dedup_question_title=False)",
        "",
        "    status = {}",
        "    for r in records:",
        "        if r['pred_text'] is not None:",
        "            key = (r['pred_text'].strip(), r['pred_label'])",
        "            status[key] = ('\\u2713' if r['pred_label'] == r['gold_label']",
        "                           else f\"\\u26a0\\ufe0f wrong label (answer key says {r['gold_label']})\")",
        "    for text, label in no_gold:",
        "        status[(text.strip(), label)] = '\\u26a0\\ufe0f doesn\\'t match the answer key'",
        "",
        "    missing = [(r['gold_label'], r['gold_text']) for r in records if r['pred_text'] is None]",
        "    return status, missing",
    ]),

    md("md-2", [
        "## Sample by sample",
    ]),
    code("walkthrough", [
        "def flag_for(text, label, status):",
        "    f = status.get((text.strip(), label), '\\u2753 not checked')",
        "    if label == 'question.text' and text[:1].isalpha() and text[:1].islower():",
        "        f += '  \\u26a0\\ufe0f starts lowercase'",
        "    return f",
        "",
        "for n in SAMPLES:",
        "    status, missing = sample_status(n)",
        "    s4 = json.loads(P.final_path(TAG, n).read_text(encoding='utf-8'))",
        "    t = s4['narrative']['template']",
        "",
        "    print()",
        "    print('=' * 90)",
        "    print(f'SAMPLE {n}')",
        "    print('=' * 90)",
        "    title = t.get('title', '')",
        "    print(f\"TITLE  {flag_for(title, 'title', status):<45} {title[:90]}\")",
        "    for s in t.get('section', []):",
        "        sec_title = s.get('title', '')",
        "        print(f\"\\nSECTION  {flag_for(sec_title, 'section.title', status):<43} {sec_title[:90]}\")",
        "        for q in s.get('question', []):",
        "            q_text = q.get('text', '')",
        "            ans = q.get('answer', {}).get('json', {}).get('answer', '')",
        "            print(f\"  Q  {flag_for(q_text, 'question.text', status):<47} {q_text[:90]}\")",
        "            print(f\"  A  {flag_for(ans, 'answer.text', status):<47} {ans[:90]}\")",
        "    if missing:",
        "        print('\\nMISSING (the answer key expects these; nothing in the document matches them):')",
        "        for label, text in missing:",
        "            print(f'  [{label}] {text[:90]}')",
    ]),

    md("md-3", [
        "## Takeaways",
        "",
        "- Most documents match the answer key cleanly, item for item.",
        "- Where something doesn't match, look at the flag: **wrong label** means the",
        "  text was found but mis-classified; **doesn't match the answer key** usually",
        "  means the document was split differently than the answer key expects (a",
        "  heading glued onto its answer, or one long answer cut into pieces);",
        "  **starts lowercase** is a quick, plain-English way to spot a question that's",
        "  actually a stray sentence fragment.",
        "- A document can also show a near-miss that looks identical to the eye but",
        "  technically doesn't match — usually a tiny extraction artifact (a",
        "  line-wrapped word like \"sub-study\" coming through with an extra space where",
        "  the PDF wrapped the line) rather than anything the model got wrong.",
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
