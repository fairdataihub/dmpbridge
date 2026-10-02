"""Build notebooks/rda-dmp-json.ipynb.

One notebook for the whole loop: the pdfplumber text of sample 14 plus the complete
maDMP 1.2 schema go to llama3.1:8b, gemma4:e4b and llama3.3:70b; the results are
scored against the hand-made reference; every saved prompt version is compared.
The machinery lives in dmpbridge/rda/, so the notebook stays short.

    python scripts/build/build_rda_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/rda-dmp-json.ipynb")


def md(cid, body):
    """Markdown cell from a plain block of text."""
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in body.strip("\n").split("\n")]}


def code(cid, body):
    """Code cell from a plain block of text."""
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in body.strip("\n").split("\n")]}


cells = [

md("title", '''
# PDF → RDA DMP JSON: run a prompt version, score it, compare versions

The text of one DMP PDF plus the complete **maDMP 1.2** schema go to three models.
What comes back is scored against the form a person filled in by hand from the same
PDF. Every prompt version is saved under its own name, so you can see what a change
to the prompt did.

**To try a new prompt:** edit Step 3, give it a new `RUN_NAME` in the settings cell,
run all cells. Step 7 shows the new version next to every earlier one.

Running again with an existing name does not call the models again — the saved results
are loaded and re-scored. If the prompt was changed but the name was not, Step 3 stops
and asks for a new name, so no version is ever overwritten.

| Step | What happens |
|---|---|
| 1 | Read sample 14 with pdfplumber |
| 2 | Load the schema — unchanged |
| 3 | The prompt — the only thing to edit between versions |
| 4 | Run the three models |
| 5 | Score this version: correct, hallucinated, missed; precision, recall, F1 |
| 6 | Every field, model by model |
| 7 | Compare every saved version |

**The marks.** Every field a model wrote is **Correct** (matches the person's value, or both
blank) or **Hallucinated** (wrong, made up, or blank where the person had a value). A field the
person filled in that the model never got right is **Missed**. Precision is the share of what
the model wrote that was right; recall is the share of the document's fields it found; F1
balances the two. Identifiers must match exactly (ignoring `https://` and case), dates by
day, a licence by name or URL, and free text counts when at least three-quarters of the
model's words appear in the person's version. The rules are in `dmpbridge/rda/score.py`.
'''),

code("setup", '''
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.rda import Version, load_schema
from dmpbridge.rda import report

RUN_NAME = "v2"                      # change this every time you change the prompt

SAMPLE    = 14
PDF       = Path("data/input/pdfs") / f"sample{SAMPLE}.pdf"
SCHEMA    = Path("data/output/rda/maDMP-schema-1.2.json")
REFERENCE = Path("data/output/rda") / f"RDA_DMP_sample{SAMPLE}_manual_annotation.json"
RUNS_DIR  = Path("data/output/rda/runs")

MODEL_1 = "llama3.1:8b"
MODEL_2 = "gemma4:e4b"
MODEL_3 = "llama3.3:70b"
MODELS  = [MODEL_1, MODEL_2, MODEL_3]

version = Version(RUN_NAME, sample=SAMPLE, models=MODELS, runs_dir=RUNS_DIR, num_ctx=32768, max_tokens=8000)
'''),

md("s1", '''
## Step 1 — Read the PDF
'''),

code("read", '''
dmp_text = get_extractor("pdfplumber").extract(PDF)[0]["text"]
print(f"sample{SAMPLE}: {len(dmp_text):,} characters")
print(dmp_text[:300])
'''),

md("s2", '''
## Step 2 — Load the schema

The schema is used exactly as published — nothing removed, nothing changed. Its `$ref`
pointers are resolved so the model sees every definition in place.
'''),

code("schema", '''
schema, schema_text, n_defs = load_schema(SCHEMA)
print(f"{len(schema_text):,} characters of schema, {n_defs} definitions resolved")
'''),

md("s3", '''
## Step 3 — The prompt

**This is the cell to edit.** The schema is given to the model twice: as text in the
prompt, and as Ollama's `format`, which constrains the JSON it generates to that
structure. The exact prompt is saved as `prompt.txt` in the version's folder.
'''),

code("prompt", '''
SYSTEM = """You convert Data Management Plans into RDA maDMP JSON.

Strict rules:
1. Use only the field names defined in the schema. Never add a key that is not in the schema.
2. Put every field exactly where the schema places it. The whole document is one top-level "dmp" object.
3. Where the schema lists allowed values, use one of them, spelled exactly as in the schema .
4. Take every value from the Data Management Plan text. Never copy example values from the schema and never hallucinate.
5. Output only the JSON object. No explanation, no markdown.
6. Include every dataset, contributor, funder, host and distribution the text mentions. Leave out any field the text gives no information for."""


def make_prompt(dmp_text):
    return f"""Here is the RDA maDMP JSON schema (version 1.2). Follow it strictly:

{schema_text}

Here is the text of a Data Management Plan:

{dmp_text}

Generate one JSON object that strictly follows the schema above, filling in
whatever information the Data Management Plan text contains. Output only the JSON."""


version.set_prompt(SYSTEM, make_prompt, schema, schema_text)
print(f"{len(make_prompt(dmp_text)):,} characters go to the model: the system text, the schema and the document")
'''),

md("s4", '''
## Step 4 — Run the three models

One model at a time: it is loaded, checked to be **100% on the GPU** (a model spilling
onto the CPU turns seconds into hours and is skipped), then called with the prompt. Each
result is saved as soon as it arrives, with the time taken and the tokens in and out.
A model that never finishes a valid JSON (it looped until the token cap) is recorded as
failed for this version, and what it wrote is kept in a `.raw.txt` file.

The 70B needs about 53 GB of VRAM at this context size — all three 24 GB cards. With
fewer cards available it cannot sit fully on the GPU and is skipped.
'''),

code("run", '''
version.run(MODEL_1, dmp_text)
'''),

code("gemma", '''
version.run(MODEL_2, dmp_text)
'''),

code("llama33", '''
version.run(MODEL_3, dmp_text)
'''),

md("s5", '''
## Step 5 — Score this version

Each model's JSON is compared field by field with the reference. One row per model:
*fields in reference* is how many fields the person filled in, *fields output* how many
the model wrote.
'''),

code("score", '''
summary, details = version.score(REFERENCE)
display(summary)
report.worked_example(summary)
'''),

code("chart", '''
report.plot_version(summary, MODELS, title=f"Version {RUN_NAME}", save_to=version.dir / "evaluation_charts.png")
'''),

md("s6", '''
## Step 6 — Every field, model by model

What each model wrote next to what the person wrote, with the mark and the reason.
Hallucinated rows come first so the problems are at the top. Missed fields are listed
in the Excel file saved with the version.
'''),

code("tables", '''
report.show_fields(details, MODELS)
'''),

md("s7", '''
## Step 7 — Compare every saved version

Every scored version under `data/output/rda/runs/`, oldest first: the scores per version
and model, F1 and recall as lines, then what changed field by field between this version
and the one before it, and the change made to the prompt.
'''),

code("compare", '''
history = report.load_history(RUNS_DIR)
report.show_history(history, MODELS)
report.plot_history(history, MODELS, save_to=RUNS_DIR / "versions.png")
'''),

code("changes", '''
report.show_changes(version, history, summary, details)
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells, "
      f"{sum(len(c['source']) for c in cells if c['cell_type'] == 'code')} lines of code")
