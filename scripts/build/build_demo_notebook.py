"""Build notebooks/demo-from-yaml-config.ipynb.

The notebook version of scripts/run_demo.py: read demo/config.yaml as the
input, run it, show the final JSON at the end. Edit demo/config.yaml, not
this notebook, to point it at a different model/extractor/sample range.

    python scripts/build/build_demo_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/demo-from-yaml-config.ipynb")


def md(cid, lines):
    """Markdown cell."""
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    """Code cell."""
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


# The labeled document rendered as colour-coded HTML. Kept as one block so the
# HTML attribute quotes don't need escaping line by line.
HTML_CELL = '''
from html import escape
from itertools import groupby
from IPython.display import HTML, display

# label -> (colour, text colour); same palette as the pipeline diagram
COLORS = {
    'title':               ('#1E406E', 'white'),
    'section.title':       ('#0F766E', 'white'),
    'section.description': ('#94A3B8', 'black'),
    'question.text':       ('#B45309', 'white'),
    'answer.text':         ('#CBD5E1', 'black'),
}
SIZE   = {'title': '20px', 'section.title': '16px'}
BOLD   = {'title', 'section.title', 'question.text'}


def pill(label, extra=''):
    bg, fg = COLORS.get(label, ('#ddd', 'black'))
    return (f"<span style='background:{bg};color:{fg};font-size:11px;padding:2px 9px;"
            f"border-radius:10px;font-family:monospace'>{escape(label)}{extra}</span>")


def group_html(label, texts):
    """One cell for a run of consecutive blocks with the same label."""
    bg, _ = COLORS.get(label, ('#ddd', 'black'))
    body = ''.join(f"<div style='margin-top:6px;white-space:pre-wrap'>{escape(x)}</div>" for x in texts)
    return (f"<div style='border-left:6px solid {bg};background:{bg}18;padding:6px 12px;"
            f"margin:6px 0;font-family:system-ui,sans-serif;"
            f"font-size:{SIZE.get(label, '13px')};"
            f"font-weight:{'bold' if label in BOLD else 'normal'}'>"
            f"{pill(label, f' x{len(texts)}' if len(texts) > 1 else '')}{body}</div>")


for n in cfg.sample_range:
    blocks = json.loads(P.labeled_path(tag, n).read_text(encoding='utf-8'))
    counts = {}
    for b in blocks:
        counts[b['label']] = counts.get(b['label'], 0) + 1
    legend = ' '.join(pill(l, f' x{c}') for l, c in counts.items())
    # consecutive blocks with the same label share one cell
    groups = [(label, [b['text'] for b in run])
              for label, run in groupby(blocks, key=lambda b: b['label'])]
    display(HTML(f"<h3 style='font-family:system-ui,sans-serif'>sample{n} - "
                 f"{len(blocks)} labeled blocks in {len(groups)} cells</h3><p>{legend}</p>"
                 + ''.join(group_html(label, texts) for label, texts in groups)))
'''


cells = [
    md("title", [
        "# Demo — from `demo/config.yaml` to the final document",
        "",
        "Input is a YAML file, not settings in this notebook — edit",
        "`demo/config.yaml` (model, extractor, fallback, sample range) and",
        "re-run this notebook top to bottom. Nothing here needs to change.",
        "",
        "**How the PDF is read.** A normal PDF goes through `pdfplumber`, which",
        "takes the text layer together with the bold, italic and underline cues",
        "its fonts carry. A PDF whose text layer is missing or broken — a scanned",
        "page, or fonts with no character-to-text mapping — fails a readability",
        "check *before* the model sees it, and is re-read from its page images by",
        "**LightOnOCR** instead (`fallback: auto`). The check is per document, so",
        "clean PDFs never touch the fallback; when one does, `[fallback]` lines",
        "below say which document and why.",
    ]),

    md("md-input", ["## Input — demo/config.yaml, as written on disk"]),
    code("input", [
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
        "from dmpbridge.evaluation.experiment import ExperimentConfig, Experiment",
        "",
        "CONFIG_PATH = Path('demo/config.yaml')",
        "print(CONFIG_PATH.read_text(encoding='utf-8'))",
    ]),

    md("md-run", ["## Run it"]),
    code("run", [
        "cfg = ExperimentConfig.from_yaml(CONFIG_PATH)",
        "exp = Experiment(cfg)",
        "exp.run()",
        "",
        "print(f'\\n{cfg.name}: {len(cfg.models)} model(s), {len(cfg.extractors)} extractor(s), '",
        "      f'samples {cfg.sample_start}-{cfg.sample_end}')",
    ]),

    md("md-output", [
        "## Output — the final document",
        "",
        "Same content `scripts/run_demo.py` copies into `demo/output/final/`; read here",
        "directly from the standard pipeline location so this always reflects the latest run.",
    ]),
    code("output", [
        "model, extractor = cfg.models[0], cfg.extractors[0]",
        "tag = cfg.tag_for(model, extractor)",
        "",
        "for n in cfg.sample_range:",
        "    final = P.final_path(tag, n)",
        "    if not final.exists():",
        "        continue",
        "    doc = json.loads(final.read_text(encoding='utf-8'))",
        "    template = doc['narrative']['template']",
        "",
        "    print(f'=== sample{n} ===')",
        "    print(f'TITLE: {template[\"title\"]}\\n')",
        "    for i, section in enumerate(template['section'], 1):",
        "        print(f'{i}. {section[\"title\"]}')",
        "        for q in section['question']:",
        "            answer = q['answer']['json']['answer']",
        "            print(f'   Q: {q[\"text\"][:70]}')",
        "            print(f'   A: {answer[:90]}{\"...\" if len(answer) > 90 else \"\"}')",
        "    print()",
    ]),

    md("md-save", [
        "## Save — copy the result into demo/output/",
        "",
        "`exp.run()` writes to the pipeline's standard location under `data/output/`.",
        "This copies each stage for the samples in the config into",
        "`demo/output/{labeled,structured,final}/` — the same layout `scripts/run_demo.py` uses.",
    ]),
    code("save", [
        "import shutil",
        "",
        "OUTPUT_DIR = Path('demo/output')",
        "STAGES = [('labeled', P.labeled_path), ('structured', P.structured_path), ('final', P.final_path)]",
        "",
        "for n in cfg.sample_range:",
        "    for stage, resolve in STAGES:",
        "        src = resolve(tag, n)",
        "        if not src.exists():",
        "            continue",
        "        dest = OUTPUT_DIR / stage / f'sample{n}.json'",
        "        dest.parent.mkdir(parents=True, exist_ok=True)",
        "        shutil.copy2(src, dest)",
        "        print(f'{stage:10} -> {dest}')",
    ]),

    md("md-html", [
        "## The labels, as a document",
        "",
        "Every block of text with the label the model gave it, colour-coded:",
        "title, section heading, section description, question, answer.",
    ]),
    code("labels-html", HTML_CELL.strip().splitlines()),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
