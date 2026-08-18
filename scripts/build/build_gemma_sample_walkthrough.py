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
        "| match | Matches the answer key |",
        "| wrong label | The text is right, but it's labelled as the wrong kind of thing |",
        "| near-miss (NN% match) — extra word(s)... | Almost identical to something in the answer key — the exact",
        "| | word(s) that differ are named, so you can judge for yourself whether it's a real |",
        "| | mistake or just a stray character (a PDF line-wrap artifact is a common cause) |",
        "| not in the answer key at all | No close match exists in the answer key — a genuine extra/spurious item |",
        "| starts lowercase | A simple warning sign for a real question: real questions start with a capital",
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
        "from dmpbridge.evaluation.evaluate import extract_gold, _match_structured, tokenize, containment",
        "from dmpbridge.evaluation.annotation_rules import resolve_new_gt_path, convert_tag_to_final",
        "",
        f"MODEL, TAG = {MODEL!r}, {TAG!r}",
        "SAMPLES = range(1, 11)",
        "NEAR_MISS_FLOOR = 0.85   # containment above this -> \"near-miss\", show the exact",
        "                         # difference; below it -> genuinely not in the answer key",
        "",
        "if not P.final_path(TAG, 1).exists():",
        "    convert_tag_to_final(TAG)",
        "",
        "",
        "def closest_gold(text, label, gold):",
        "    \"\"\"Best-matching gold item of the same label, by word containment, or",
        "    (None, 0.0) if this label has no gold items at all.\"\"\"",
        "    pt = tokenize(text)",
        "    best_text, best_score = None, 0.0",
        "    for gtext, glabel in gold:",
        "        if glabel != label:",
        "            continue",
        "        score = containment(pt, tokenize(gtext))",
        "        if score > best_score:",
        "            best_text, best_score = gtext, score",
        "    return best_text, best_score",
        "",
        "",
        "def sample_status(n):",
        "    \"\"\"(status_lookup, missing) for one sample's final JSON vs its answer key.",
        "",
        "    status_lookup maps (text, label) -> a short flag string, built from every",
        "    predicted item; missing lists answer-key items nothing in the document",
        "    matched. Path B only: dedup_question_title=False is Path B's own setting,",
        "    since it exists specifically to measure the fill-in step Path A skips.",
        "",
        "    A predicted item that doesn't match anything gets one of two different",
        "    flags, not the same generic one: if it's a near-miss of some gold item",
        "    (very high word overlap, just not 100%), the flag names the exact word(s)",
        "    that differ; only a predicted item with no close gold counterpart at all",
        "    gets the plain \"not in the answer key\" flag.",
        "    \"\"\"",
        "    gold = extract_gold(resolve_new_gt_path(n), dedup_question_title=False)",
        "    records, no_gold = _match_structured(P.final_path(TAG, n), gold, dedup_question_title=False)",
        "",
        "    status = {}",
        "    for r in records:",
        "        if r['pred_text'] is not None:",
        "            key = (r['pred_text'].strip(), r['pred_label'])",
        "            status[key] = ('match' if r['pred_label'] == r['gold_label']",
        "                           else f\"wrong label (answer key says {r['gold_label']})\")",
        "    for text, label in no_gold:",
        "        gold_text, score = closest_gold(text, label, gold)",
        "        if gold_text is None or score < NEAR_MISS_FLOOR:",
        "            status[(text.strip(), label)] = 'not in the answer key at all'",
        "            continue",
        "        missing_words = tokenize(text) - tokenize(gold_text)",
        "        if missing_words:",
        "            status[(text.strip(), label)] = (",
        "                f\"near-miss ({score:.0%} match) \\u2014 \"",
        "                f\"extra word(s) not in the answer key: {sorted(missing_words)}\")",
        "        else:",
        "            # Every word IS in the answer key, but this exact answer-key item was",
        "            # already claimed by a different, better-fitting piece of the",
        "            # prediction \\u2014 so this is leftover/duplicate content, not a word",
        "            # mismatch (e.g. one long answer got split into overlapping pieces).",
        "            status[(text.strip(), label)] = (",
        "                'duplicate content \\u2014 these exact words are already '",
        "                'matched by a different part of the document above')",
        "",
        "    missing = [(r['gold_label'], r['gold_text']) for r in records if r['pred_text'] is None]",
        "    return status, missing",
    ]),

    md("md-2", [
        "## Sample by sample",
    ]),
    code("walkthrough", [
        "def flag_for(text, label, status):",
        "    f = status.get((text.strip(), label), 'not checked')",
        "    if label == 'question.text' and text[:1].isalpha() and text[:1].islower():",
        "        f += '  starts lowercase'",
        "    return f",
        "",
        "def show_item(tag, text, label, status):",
        "    \"\"\"Print one item. Short flags (just 'match') stay on the text's own",
        "    line; long ones (a near-miss explanation) print on the line above instead",
        "    of being crammed into a fixed-width column, so nothing gets truncated.\"\"\"",
        "    flag = flag_for(text, label, status)",
        "    if flag == 'match':",
        "        print(f'  {tag}  match  {text[:90]}')",
        "    else:",
        "        print(f'  {tag}  {flag}')",
        "        print(f'  {\" \" * len(tag)}     {text[:110]}')",
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
        "    show_item('TITLE  ', t.get('title', ''), 'title', status)",
        "    for s in t.get('section', []):",
        "        print()",
        "        show_item('SECTION', s.get('title', ''), 'section.title', status)",
        "        for q in s.get('question', []):",
        "            q_text = q.get('text', '')",
        "            ans = q.get('answer', {}).get('json', {}).get('answer', '')",
        "            show_item('Q      ', q_text, 'question.text', status)",
        "            show_item('A      ', ans, 'answer.text', status)",
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
