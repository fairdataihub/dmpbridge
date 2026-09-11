"""Build notebooks/pdf-to-rda-dmp-json.ipynb.

One prompt: the pdfplumber text of a DMP plus the complete, unmodified maDMP 1.2
schema, with strict instructions to follow the schema. Run with llama3.1:8b, then
the same prompt with gemma4:e4b and llama3.3:70b, and save all three results.

    python scripts/build/build_rda_extraction_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/pdf-to-rda-dmp-json.ipynb")


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
# PDF to RDA DMP JSON

One prompt: the text of a DMP PDF plus the complete **maDMP 1.2** schema, with
strict instructions to follow the schema. Run first with `llama3.1:8b`, then the
same prompt with `gemma4:e4b` and `llama3.3:70b`.

| Step | What happens |
|---|---|
| 1 | Read sample 14 with pdfplumber |
| 2 | Load the schema — unchanged |
| 3 | Prompt → `llama3.1:8b` |
| 4 | Same prompt → `gemma4:e4b` |
| 5 | Same prompt → `llama3.3:70b` |
| 6 | Save all three |
'''),

code("setup", '''
import json
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.models.ollama import OllamaModel

PDF     = Path("data/input/pdfs/sample14.pdf")
SCHEMA  = Path("data/output/rda/maDMP-schema-1.2.json")
HOST    = "http://localhost:11434"
MODEL_1 = "llama3.1:8b"
MODEL_2 = "gemma4:e4b"
MODEL_3 = "llama3.3:70b"
OUT_DIR = Path("data/output/rda")
'''),

md("s1", '''
## Step 1 — Read the PDF
'''),

code("read", '''
dmp_text = get_extractor("pdfplumber").extract(PDF)[0]["text"]

print(f"{len(dmp_text):,} characters\\n")
print(dmp_text[:500])
'''),

md("s2", '''
## Step 2 — Load the schema

The schema is used exactly as published — nothing removed, nothing changed. Its
`$ref` pointers are resolved to the definitions they point to, so the model sees
the schema in one piece instead of copying `"$ref"` into its answer.
'''),

code("schema", '''
schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
defs = schema["$defs"]


def resolve_refs(node):
    """Replace every $ref with the definition it points at. Content unchanged."""
    if isinstance(node, dict):
        if "$ref" in node:
            return resolve_refs(defs[node["$ref"].split("/")[-1]])
        return {k: resolve_refs(v) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [resolve_refs(v) for v in node]
    return node


schema_full = resolve_refs(schema)
schema_text = json.dumps(schema_full, separators=(",", ":"))

print(f"{len(schema_text):,} characters of schema, {len(defs)} definitions resolved")
'''),

md("s3", '''
## Step 3 — Prompt → llama3.1:8b

The schema is given to the model twice: as text in the prompt, and as Ollama's
`format`, which constrains the JSON it generates to that structure. Without the
constraint, `llama3.1:8b` fell into a loop on this prompt and never stopped;
`num_predict` is a cap in case it ever does again.
'''),

code("prompt", '''
SYSTEM = """You convert Data Management Plans into RDA maDMP JSON.

Strict rules:
1. Use only the field names defined in the schema. Never add a key that is not in the schema.
2. Put every field exactly where the schema places it. The whole document is one top-level "dmp" object.
3. Where the schema lists allowed values, use one of them, spelled exactly as in the schema.
4. Take every value from the Data Management Plan text. Never copy example values from the schema.
5. Output only the JSON object. No explanation, no markdown."""

PROMPT = f"""Here is the RDA maDMP JSON schema (version 1.2). Follow it strictly:

{schema_text}

Here is the text of a Data Management Plan:

{dmp_text}

Generate one JSON object that strictly follows the schema above, filling in
whatever information the Data Management Plan text contains. Output only the JSON."""


def run(model):
    """Send the prompt to one model; the schema is also the output constraint."""
    llm = OllamaModel(model=model, host=HOST, num_ctx=32768, num_predict=8000)
    return json.loads(llm.complete(SYSTEM, PROMPT, schema=schema_full))


result_llama = run(MODEL_1)
print(json.dumps(result_llama, indent=2, ensure_ascii=False))
'''),

md("s4", '''
## Step 4 — Same prompt → gemma4:e4b
'''),

code("gemma", '''
result_gemma = run(MODEL_2)
print(json.dumps(result_gemma, indent=2, ensure_ascii=False))
'''),

md("s5", '''
## Step 5 — Same prompt → llama3.3:70b

The 70B needs about 42 GB of VRAM, so the two smaller models are unloaded first.
This call takes a few minutes.
'''),

code("llama33", '''
import subprocess
for m in (MODEL_1, MODEL_2):
    subprocess.run(["ollama", "stop", m], check=False)

result_llama33 = run(MODEL_3)
print(json.dumps(result_llama33, indent=2, ensure_ascii=False))
'''),

md("s6", '''
## Step 6 — Save all three
'''),

code("save", '''
OUT_DIR.mkdir(parents=True, exist_ok=True)
for model, result in ((MODEL_1, result_llama), (MODEL_2, result_gemma),
                      (MODEL_3, result_llama33)):
    out = OUT_DIR / f"{PDF.stem}.rda.{model.replace(':', '-')}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{model:14} -> {out}")
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
