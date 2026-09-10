"""Build notebooks/pdf-to-rda-dmp-json.ipynb.

PDF to RDA DMP JSON: take the pdfplumber blob text of one DMP, hand it to
llama3.1:8b together with the official RDA maDMP 1.2 JSON schema, and ask for a
JSON object that complies with the schema.

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

Take one Data Management Plan PDF and ask `llama3.1:8b` to turn it into JSON that
complies with the **RDA DMP Common Standard**.

The whole idea is a single prompt: here is the schema, here is the document,
extract whatever you can.

| Step | What happens |
|---|---|
| 1 | Read the PDF into text with pdfplumber |
| 2 | Clean the text |
| 3 | Load the RDA schema |
| 4 | One prompt — schema + document, JSON out |
| 5 | Save it |

Schema: [`maDMP-schema-1.2.json`](../maDMP-schema-1.2.json), from
[RDA-DMP-Common-Standard](https://github.com/RDA-DMP-Common/RDA-DMP-Common-Standard/blob/master/examples/JSON/JSON-schema/1.2/maDMP-schema-1.2.json).

Run the cells top to bottom. Only the settings in the next cell need changing.
'''),

code("setup", '''
import json
import re
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.models.ollama import OllamaModel

# ── Settings ──────────────────────────────────────────────────────────────────
PDF    = Path("data/input/pdfs/sample14.pdf")
MODEL  = "llama3.1:8b"
SCHEMA = Path("maDMP-schema-1.2.json")
OUT    = Path("data/output/rda") / (PDF.stem + ".rda.json")

print("PDF    :", PDF)
print("Model  :", MODEL)
print("Schema :", SCHEMA)
print("Output :", OUT)
'''),

md("s1", '''
## Step 1 — Read the PDF

The project's pdfplumber extractor returns the whole document as one blob of text.
'''),

code("read", '''
raw_text = get_extractor("pdfplumber").extract(PDF)[0]["text"]

print(f"{len(raw_text):,} characters, {len(raw_text.split()):,} words\\n")
print(raw_text[:500])
'''),

md("s2", '''
## Step 2 — Clean the text

pdfplumber wraps visually emphasized words in `**bold**`, `_italic_` and
`++underline++` markers. Those are useful elsewhere in this project but are just
noise here, so we strip them and tidy the spacing. No words are changed.
'''),

code("clean", '''
def clean_text(text):
    """Strip emphasis markers and normalize whitespace."""
    t = re.sub(r"\\+\\+(.+?)\\+\\+", r"\\1", text)
    t = re.sub(r"\\*\\*(.+?)\\*\\*", r"\\1", t)
    t = re.sub(r"(?<!\\w)_(.+?)_(?!\\w)", r"\\1", t)
    t = re.sub(r"[ \\t]+", " ", t)
    t = re.sub(r"\\n{3,}", "\\n\\n", t)
    return t.strip()


clean = clean_text(raw_text)

print(f"{len(raw_text):,} chars -> {len(clean):,} chars\\n")
print(clean[:500])
'''),

md("s3", '''
## Step 3 — Load the RDA schema

**One thing has to be done to the schema before it goes in the prompt.** It is
written with `$ref` pointers into a `$defs` section, like
`"contact": {"$ref": "#/$defs/Contact"}`. Handed over as-is, the model copies
those pointers straight into its answer, and you get
`"contact": {"$ref": "#/$defs/Contact"}` back instead of an actual contact.

So we replace every `$ref` with the definition it points at. The schema has no
circular references, so this flattens cleanly.
'''),

code("schema", '''
schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
defs = schema.get("$defs", {})


def inline_refs(node, depth=0):
    """Replace every $ref with the definition it points at."""
    if depth > 25:
        return {}
    if isinstance(node, dict):
        if "$ref" in node:
            return inline_refs(defs.get(node["$ref"].split("/")[-1], {}), depth + 1)
        return {k: inline_refs(v, depth) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [inline_refs(v, depth) for v in node]
    return node


flat_schema = inline_refs({k: v for k, v in schema.items() if k != "$defs"})
schema_text = json.dumps(flat_schema, separators=(",", ":"))

print(f"as downloaded : {len(json.dumps(schema)):,} chars, {len(defs)} definitions")
print(f"after inlining: {len(schema_text):,} chars")
print(f'contains $ref : {"$ref" in schema_text}')
print()
print(f"schema   ~{len(schema_text)//4:,} tokens")
print(f"document ~{len(clean)//4:,} tokens")
print(f"prompt   ~{(len(schema_text)+len(clean))//4:,} tokens   (context is 32,768)")
'''),

md("s4", '''
## Step 4 — One prompt

Schema in, document in, JSON out. The model is told to leave out anything the
document does not state, rather than filling the gaps with plausible-looking
values.
'''),

code("ask", '''
llm = OllamaModel(model=MODEL, host="http://localhost:11434", num_ctx=32768)

SYSTEM = """You convert Data Management Plan documents into JSON matching the RDA
DMP Common Standard schema you are given.

Use only information written in the document. Leave out anything it does not
state. Never invent identifiers, emails, ORCIDs, DOIs, grant numbers or dates.
Never output "$ref" - write real values. Output JSON only."""

PROMPT = f"""Here is the RDA DMP Common Standard JSON schema (version 1.2):

{schema_text}

Here is the Data Management Plan:

--- DOCUMENT ---
{clean}
--- END DOCUMENT ---

Produce one JSON object that follows the schema above, filling in everything the
document actually states and leaving out everything it does not.

The whole result goes inside a single top-level "dmp" object, and every field must
be nested exactly where the schema puts it. Output only the JSON object."""

raw = llm.complete(SYSTEM, PROMPT, schema="json")   # Ollama JSON mode

try:
    result = json.loads(raw)
except json.JSONDecodeError:
    m = re.search(r"\\{.*\\}", raw, re.S)
    result = json.loads(m.group(0)) if m else {}

print(f"{len(raw):,} characters returned")
print("top level :", list(result))
print("dmp fields:", list(result.get("dmp", {})))
'''),

md("s4b", '''
### What came back
'''),

code("show", '''
print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])
'''),

md("s5", '''
## Step 5 — Save it
'''),

code("save", '''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"saved -> {OUT}  ({OUT.stat().st_size:,} bytes)")
'''),

md("check", '''
---

### Quick sanity check

Not part of the pipeline — just a look at whether the identifiers in the output
are actually in the PDF. Emails, ORCIDs and DOIs are the values a model is most
likely to produce from thin air, and the hardest to catch by eye, because an
invented one looks exactly like a real one.

Delete this cell if you don't want it.
'''),

code("sanity", '''
blob = json.dumps(result)
found  = set(re.findall(r"[\\w.+-]+@[\\w.-]+\\.\\w+", blob))
found |= set(re.findall(r"\\d{4}-\\d{4}-\\d{4}-[\\dX]{4}", blob))
found |= set(re.findall(r"10\\.\\d{4,}/[^\\s\\",]+", blob))

low = clean.lower()
if not found:
    print("no emails, ORCIDs or DOIs in the output")
for value in sorted(found):
    ok = value.lower() in low
    print(f"  {value:46} {'in the PDF' if ok else 'NOT IN THE PDF'}")
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
