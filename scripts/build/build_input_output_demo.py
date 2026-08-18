"""Build notebooks/demo-input-output.ipynb.

The smallest possible notebook: pick one sample, show what goes in, show
what comes out. No evaluation, no charts, no scores — just input and
output, for someone who has never seen this pipeline before.

    python scripts/build/build_input_output_demo.py
"""
import json
from pathlib import Path

NB = Path("notebooks/demo-input-output.ipynb")


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
        "# Input → output, one document",
        "",
        "What a user actually gets: point the pipeline at one sample number, and",
        "compare what went in against what came out. No scoring, no charts — this",
        "notebook only answers \"what does the input look like, what does the output",
        "look like\".",
    ]),

    md("md-settings", [
        "## Pick a sample",
        "",
        "`dmpbridge` is installed as a real Python package (`pip install -e .`) —",
        "importable from any directory, not just this project folder. Everything below",
        "is imported from it, the same as any other installed library.",
    ]),
    code("settings", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import json",
        "",
        "import dmpbridge",
        "print(f'dmpbridge {dmpbridge.__version__}, installed at {Path(dmpbridge.__file__).parent}')",
        "",
        "from dmpbridge.core import paths as P",
        "",
        "SAMPLE    = 1                  # <- change this to look at a different document",
        "MODEL     = 'llama3.1:8b'",
        "EXTRACTOR = 'pdfplumber'",
        "",
        "tag = P.make_tag(MODEL, EXTRACTOR)",
        "pdf_path = Path('data/input/pdfs') / f'sample{SAMPLE}.pdf'",
        "print(f'PDF      : {pdf_path}')",
        "print(f'model    : {MODEL}')",
        "print(f'extractor: {EXTRACTOR}')",
    ]),

    # ── input ──────────────────────────────────────────────────────────
    md("md-input", [
        "## Input",
        "",
        "The pipeline never reads the PDF's meaning directly — the first step turns it",
        "into a flat list of text blocks. This is the actual input the model sees, before",
        "any labeling happens.",
    ]),
    code("input", [
        "extracted_path = P.extracted_path(EXTRACTOR, SAMPLE)",
        "blocks = json.loads(extracted_path.read_text(encoding='utf-8'))",
        "",
        "print(f'{len(blocks)} text blocks extracted from {pdf_path.name}\\n')",
        "for b in blocks[:6]:",
        "    tag_mark = '[heading]' if b.get('is_bold') else '         '",
        "    print(f\"  {tag_mark} {b['text'][:80]!r}\")",
        "if len(blocks) > 6:",
        "    print(f'  ... and {len(blocks) - 6} more blocks')",
    ]),

    # ── output ─────────────────────────────────────────────────────────
    md("md-output", [
        "## Output",
        "",
        "After labeling, structuring, and filling in blank questions from their section",
        "heading, this is the final document — the thing a user actually opens.",
    ]),
    code("output", [
        "final_path = P.final_path(tag, SAMPLE)",
        "doc = json.loads(final_path.read_text(encoding='utf-8'))",
        "template = doc['narrative']['template']",
        "",
        "print(f'TITLE: {template[\"title\"]}\\n')",
        "for i, section in enumerate(template['section'], 1):",
        "    print(f'{i}. {section[\"title\"]}')",
        "    if section.get('description'):",
        "        print(f'   description: {section[\"description\"][:80]}')",
        "    for q in section['question']:",
        "        answer = q['answer']['json']['answer']",
        "        print(f'   Q: {q[\"text\"][:80]}')",
        "        print(f'   A: {answer[:100]}{\"...\" if len(answer) > 100 else \"\"}')",
        "    print()",
    ]),

    md("md-close", [
        "That's the whole transformation: a PDF becomes a flat list of text blocks",
        "(input), and after labeling, structuring, and rule-filling, becomes a nested",
        "title/section/question/answer document (output). Change `SAMPLE`, `MODEL`, or",
        "`EXTRACTOR` above and re-run to see a different document.",
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
