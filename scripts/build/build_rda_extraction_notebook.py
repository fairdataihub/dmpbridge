"""Build notebooks/pdf-to-rda-dmp-json.ipynb.

One simple prompt: the pdfplumber text of a DMP plus the complete maDMP 1.2
schema, asking llama3.1:8b for JSON that complies with the schema.

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

One simple prompt. Give `llama3.1:8b` the text of a DMP PDF and the complete
**maDMP 1.2** schema, and ask for JSON that complies with the schema, extracting
whatever information the DMP contains.

| Step | What happens |
|---|---|
| 1 | Read sample 14 with pdfplumber |
| 2 | Load the schema |
| 3 | Prompt: schema + DMP text → JSON |
| 4 | Save |
'''),

code("setup", '''
import json
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.models.ollama import OllamaModel

PDF    = Path("data/input/pdfs/sample14.pdf")
SCHEMA = Path("data/output/rda/maDMP-schema-1.2.json")
MODEL  = "llama3.1:8b"
OUT    = Path("data/output/rda") / (PDF.stem + ".rda.json")
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

The schema uses `$ref` pointers into its `$defs` section. They are replaced with
the definitions they point to, so the model sees the complete schema in one piece.
'''),

code("schema", '''
schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
defs = schema["$defs"]


def inline_refs(node):
    """Replace every $ref with the definition it points at."""
    if isinstance(node, dict):
        if "$ref" in node:
            return inline_refs(defs[node["$ref"].split("/")[-1]])
        return {k: inline_refs(v) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [inline_refs(v) for v in node]
    return node


def without(node, keys, in_properties=False):
    """Drop the given annotation keys — but never a field NAME: dmp.title,
    dataset.description and distribution.format are real fields that happen
    to share a name with schema annotations."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if in_properties:
                out[k] = without(v, keys)
            elif k == "properties":
                out[k] = without(v, keys, in_properties=True)
            elif k not in keys:
                out[k] = without(v, keys)
        return out
    if isinstance(node, list):
        return [without(v, keys) for v in node]
    return node


full = inline_refs(schema)

# For the prompt: the complete schema, minus its "examples" — the model copied
# example values (a sample funder ID, a sample grant URL) into the output as
# if the DMP had stated them.
schema_text = json.dumps(without(full, {"examples"}), separators=(",", ":"))

# For Ollama's `format`: the structure only. Ollama constrains decoding to it,
# which is what makes the output follow the schema — and stop.
format_schema = without(full, {"description", "title", "examples", "format", "$schema", "$id"})

print(f"{len(schema_text):,} characters of schema in the prompt")
print(f"{len(json.dumps(format_schema)):,} characters of schema as the output constraint")
'''),

md("s3", '''
## Step 3 — Prompt

The schema is given to the model twice: as text in the prompt, and as Ollama's
`format`, which constrains the JSON it generates to that structure. Without the
constraint, `llama3.1:8b` fell into a loop on this prompt — the same contributor
block repeated 47 times — and never stopped. `num_predict` is a cap in case it
ever does again.

One consequence to know about: the schema **requires** `contact.mbox`, `created`
and `modified`. This DMP states none of them, so the model fills them in itself.
'''),

code("prompt", '''
llm = OllamaModel(model=MODEL, host="http://localhost:11434",
                  num_ctx=32768, num_predict=8000)

PROMPT = f"""Here is the RDA maDMP JSON schema (version 1.2):

{schema_text}

Here is the text of a Data Management Plan:

{dmp_text}

Generate a JSON document that complies with the schema above, extracting whatever
information is possible from the Data Management Plan text. Output only the JSON."""

raw = llm.complete("You convert Data Management Plans into RDA maDMP JSON.",
                   PROMPT, schema=format_schema)
result = json.loads(raw)

print(json.dumps(result, indent=2, ensure_ascii=False))
'''),

md("s4", '''
## Step 4 — Save
'''),

code("save", '''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"saved -> {OUT}")
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
