"""Build notebooks/10-demo-from-yaml.ipynb.

The notebook version of scripts/run_demo.py: read demo/config.yaml as the
input, run it, show the final JSON at the end. Edit demo/config.yaml, not
this notebook, to point it at a different model/extractor/sample range.

    python scripts/build/build_demo_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/10-demo-from-yaml.ipynb")


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
        "# Demo — from `demo/config.yaml` to the final document",
        "",
        "Input is a YAML file, not settings in this notebook — edit",
        "`demo/config.yaml` (model, extractor, sample range) and re-run this",
        "notebook top to bottom. Nothing here needs to change.",
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
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
