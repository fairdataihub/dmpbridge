"""Build notebooks/demo-from-yaml-config.ipynb.

The notebook version of scripts/run_demo.py: read demo/config.yaml as the
input, run it - the narrative stages 1-4, then with an ``rda:`` block the RDA
stages 5-6 - and show the results. Edit demo/config.yaml, not this notebook, to
point it at a different model/extractor/sample range.

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
    # The final document (stage 4) as saved: sections -> questions -> answers.
    final = json.loads(P.final_path(tag, n).read_text(encoding='utf-8'))['narrative']['template']
    parts = [group_html('title', [final['title']])] if final.get('title') else []
    n_questions = 0
    for s in final['section']:
        parts.append(group_html('section.title', [s['title']]))
        if s.get('description'):
            parts.append(group_html('section.description', [s['description']]))
        for q in s.get('question', []):
            n_questions += 1
            parts.append(group_html('question.text', [q.get('text', '')]))
            answer = q.get('answer', {}).get('json', {}).get('answer', '')
            if answer:
                parts.append(group_html('answer.text', [answer]))
    display(HTML(f"<h3 style='font-family:system-ui,sans-serif'>sample{n} - "
                 f"the final document: {len(final['section'])} sections, {n_questions} questions</h3>"
                 + ''.join(parts)))
'''


cells = [
    md("title", [
        "# Demo — from `demo/config.yaml` to the final document",
        "",
        "Input is a YAML file, not settings in this notebook — edit",
        "`demo/config.yaml` (narrative model, RDA model, extractor, fallback, sample range) and",
        "re-run this notebook top to bottom. Nothing here needs to change.",
        "",
        "**The six stages.** Each writes its own folder under `data/output/`:",
        "",
        "| stage | folder | what it holds |",
        "|---|---|---|",
        "| 1 | `1_extracted/` | the plan's text |",
        "| 2 | `2_labeled/` | the text, labeled by the narrative model |",
        "| 3 | `3_structured/` | nested into the DMP Tool schema |",
        "| 4 | `4_final/` | Rules.xlsx conversion applied — the narrative JSON |",
        "| 5 | `5_rda/` | the RDA maDMP 1.2 JSON, from the same text, by the RDA model |",
        "| 6 | `6_merged/` | `{\"dmp\": <stage 5>, \"narrative\": <stage 4>}` |",
        "",
        "Stages 5 and 6 run only when the config has an `rda:` block.",
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
        "from dmpbridge.core.pipeline import normalize_fallbacks",
        "from dmpbridge.evaluation.experiment import Experiment",
        "from dmpbridge.rda.config import load_pipeline_config",
        "from dmpbridge.rda.stages import run_rda_stages",
        "from dmpbridge.strategies.wholedoc import WholeDocStrategy",
        "",
        "CONFIG_PATH = Path('demo/config.yaml')",
        "print(CONFIG_PATH.read_text(encoding='utf-8'))",
    ]),

    md("md-run", [
        "## Run it",
        "",
        "The `narrative:` block's model runs stages 1-4 through the experiment runner;",
        "the `rda:` block's model runs stages 5-6 after it, from the same stage-1 text.",
    ]),
    code("run", [
        "cfg, rda_model = load_pipeline_config(CONFIG_PATH)",
        "",
        "# Stages 1-4: the narrative pipeline.",
        "Experiment(cfg).run()",
        "",
        "# Stages 5-6: the RDA JSON, merged with the narrative JSON.",
        "if rda_model:",
        "    for extractor in cfg.extractors:",
        "        strategy = WholeDocStrategy(model=rda_model, host=cfg.host, extractor=extractor,",
        "                                    cache_dir=P.EXTRACTED_DIR / extractor,",
        "                                    fallback_extractors=normalize_fallbacks(cfg.fallback, extractor))",
        "        for model in cfg.models:",
        "            for n in cfg.sample_range:",
        "                run_rda_stages(strategy, Path(cfg.pdf_dir) / f'sample{n}.pdf', n,",
        "                               rda_model, model, extractor, host=cfg.host)",
        "",
        "print(f'\\n{cfg.name}: narrative model {\", \".join(cfg.models)}, RDA model {rda_model or \"(none)\"}, '",
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

    md("md-rda", [
        "## Output — the RDA JSON and the merged file",
        "",
        "Stage 5 (what the RDA model filled in) and the top level of stage 6.",
    ]),
    code("rda", [
        "if not rda_model:",
        "    print('No rda: block in demo/config.yaml - stages 5-6 not run.')",
        "else:",
        "    rda_tag = P.rda_tag(rda_model, extractor)",
        "    merged_tag = P.merged_tag(model, rda_model, extractor)",
        "    for n in cfg.sample_range:",
        "        path = P.rda_path(rda_tag, n)",
        "        if not path.exists():",
        "            print(f'=== sample{n}: no RDA JSON (see the run log above) ===')",
        "            continue",
        "        dmp = json.loads(path.read_text(encoding='utf-8'))['dmp']",
        "        contact = dmp.get('contact', {})",
        "        ids = ', '.join(i.get('identifier', '') for i in contact.get('contact_id', []))",
        "        print(f'=== sample{n} - {rda_model} ===')",
        "        print(f'TITLE    : {dmp.get(\"title\", \"\")}')",
        "        print(f'CONTACT  : {contact.get(\"name\") or \"-\"}  {contact.get(\"mbox\") or \"\"}  {ids}')",
        "        print(f'LANGUAGE : {dmp.get(\"language\", \"\")}   ETHICAL ISSUES: {dmp.get(\"ethical_issues_exist\", \"\")}')",
        "        print(f'CREATED  : {dmp.get(\"created\", \"\")}   MODIFIED: {dmp.get(\"modified\", \"\")}')",
        "        datasets = dmp.get('dataset', [])",
        "        print(f'DATASETS : {len(datasets)}')",
        "        for d in datasets:",
        "            print(f'   - {d.get(\"title\", \"\")}  (personal data: {d.get(\"personal_data\", \"\")}, '",
        "                  f'sensitive: {d.get(\"sensitive_data\", \"\")}, '",
        "                  f'{len(d.get(\"distribution\", []))} distribution(s))')",
        "        for key in ('contributor', 'project', 'cost'):",
        "            print(f'{key.upper():9}: {len(dmp.get(key, []))}')",
        "",
        "        merged = P.merged_path(merged_tag, n)",
        "        if merged.exists():",
        "            print(f'\\nmerged   : {merged}')",
        "            print(f'top level: {list(json.loads(merged.read_text(encoding=\"utf-8\")))}')",
        "        print()",
    ]),

    md("md-rda-json", [
        "## The RDA JSON",
        "",
        "Stage 5 in full, exactly as saved in `5_rda/` — the `dmp` part of the merged file.",
    ]),
    code("rda-json", [
        "if rda_model:",
        "    for n in cfg.sample_range:",
        "        path = P.rda_path(P.rda_tag(rda_model, extractor), n)",
        "        if path.exists():",
        "            print(f'=== sample{n} - {path} ===')",
        "            print(json.dumps(json.loads(path.read_text(encoding='utf-8')), indent=2, ensure_ascii=False))",
    ]),

    md("md-save", [
        "## Save — copy the result into demo/output/",
        "",
        "The run writes to the pipeline's standard location under `data/output/`.",
        "This copies each stage for the samples in the config into",
        "`demo/output/{extracted,labeled,structured,final,rda,merged}/` — the same layout",
        "`scripts/run_demo.py` uses.",
    ]),
    code("save", [
        "import shutil",
        "",
        "OUTPUT_DIR = Path('demo/output')",
        "",
        "for n in cfg.sample_range:",
        "    stages = [('extracted', P.extracted_path(extractor, n)),",
        "              ('labeled', P.labeled_path(tag, n)),",
        "              ('structured', P.structured_path(tag, n)),",
        "              ('final', P.final_path(tag, n))]",
        "    if rda_model:",
        "        stages += [('rda', P.rda_path(P.rda_tag(rda_model, extractor), n)),",
        "                   ('merged', P.merged_path(P.merged_tag(model, rda_model, extractor), n))]",
        "    for stage, src in stages:",
        "        if not src.exists():",
        "            continue",
        "        dest = OUTPUT_DIR / stage / f'sample{n}.json'",
        "        dest.parent.mkdir(parents=True, exist_ok=True)",
        "        shutil.copy2(src, dest)",
        "        print(f'{stage:10} -> {dest}')",
    ]),

    md("md-html", [
        "## The final document",
        "",
        "The document as saved in `demo/output/final/`, colour-coded by label and",
        "arranged as sections, questions and answers.",
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
