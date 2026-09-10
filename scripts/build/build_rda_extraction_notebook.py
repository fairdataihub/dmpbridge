"""Build notebooks/pdf-to-rda-dmp-json.ipynb.

PDF to RDA DMP JSON. Builds a complete skeleton from the official maDMP 1.2
schema — every key the standard defines, all null — then asks llama3.1:8b to fill
in whatever the DMP document actually says. Anything the document does not
mention stays null, so the output always has the schema's full shape.

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

Turn one Data Management Plan PDF into **RDA DMP Common Standard** JSON with
`llama3.1:8b`, keeping the standard's complete structure.

**How this works.** The skeleton is built from the schema itself — every key the
standard defines, in the right place, set to `null`. The model is then asked only
to supply *values*. Whatever it finds gets written in; whatever the document does
not mention stays `null`.

That means the output always has the full schema shape, and fields can never land
in the wrong place, because we build the structure rather than asking the model to
get the nesting right.

| Step | What happens |
|---|---|
| 1 | Read the PDF into text with pdfplumber |
| 2 | Clean the text |
| 3 | Build the empty skeleton from the schema |
| 4 | Ask the model for the values |
| 5 | Fill the skeleton — everything else stays null |
| 6 | Check it against the schema |
| 7 | Save it |

Schema: `maDMP-schema-1.2.json`, from
[RDA-DMP-Common-Standard](https://github.com/RDA-DMP-Common/RDA-DMP-Common-Standard/blob/master/examples/JSON/JSON-schema/1.2/maDMP-schema-1.2.json).
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
SCHEMA = Path("data/output/rda/maDMP-schema-1.2.json")
OUT    = Path("data/output/rda") / (PDF.stem + ".rda.json")

print("PDF    :", PDF)
print("Model  :", MODEL)
print("Schema :", SCHEMA)
print("Output :", OUT)
'''),

md("s1", '''
## Step 1 — Read the PDF
'''),

code("read", '''
raw_text = get_extractor("pdfplumber").extract(PDF)[0]["text"]

print(f"{len(raw_text):,} characters, {len(raw_text.split()):,} words\\n")
print(raw_text[:400])
'''),

md("s2", '''
## Step 2 — Clean the text

Strips the `**bold**` / `_italic_` / `++underline++` markers pdfplumber adds and
tidies the spacing. No words are changed.
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
print(clean[:400])
'''),

md("s3", '''
## Step 3 — Build the empty skeleton

Walk the schema and produce every key it defines, set to `null`. Two things have
to be handled on the way:

- **`$ref`** — the schema keeps 48 definitions in `$defs` and points at them, so
  each pointer is replaced with what it points to.
- **`oneOf`** — some fields accept either one object or a list of them
  (`contact_id` is written this way). We take the first form.

Arrays get one specimen entry, so you can see the shape of what belongs there.
'''),

code("skeleton", '''
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


flat = inline_refs({k: v for k, v in schema.items() if k != "$defs"})


def build_skeleton(node, depth=0):
    """Every key the schema defines; null at the leaves, one entry per array."""
    if depth > 12 or not isinstance(node, dict):
        return None
    if "oneOf" in node or "anyOf" in node:
        # "either one object or a list of them" (contact_id, contributor_id,
        # creator_id, metadata_standard_id). Take the list form: it is what the
        # RDA skeleton uses, and it keeps every sub-key the schema defines.
        options = [o for o in (node.get("oneOf") or node.get("anyOf"))
                   if o.get("type") != "null"]
        pick = next((o for o in options if o.get("type") == "array"), options[0])
        return build_skeleton(pick, depth)
    kind = node.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    if kind == "object" or "properties" in node:
        return {k: build_skeleton(v, depth + 1)
                for k, v in node.get("properties", {}).items()}
    if kind == "array":
        return [build_skeleton(node.get("items", {}), depth + 1)]
    return None


SKELETON = build_skeleton(flat)


def leaf_paths(obj, prefix=""):
    """Dotted path for every leaf; '[]' marks a list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaf_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        if obj:
            yield from leaf_paths(obj[0], prefix + "[]")
    else:
        yield prefix


ALL_PATHS = list(leaf_paths(SKELETON))

print(f"{len(ALL_PATHS)} fields in the skeleton\\n")
print("top-level dmp keys:")
for k in SKELETON["dmp"]:
    print("  ", k)
'''),

md("s4", '''
## Step 4 — Ask the model for the values

109 fields is too many for one call, so they are asked for a subtree at a time —
`contact`, `contributor`, `dataset`, `project` and so on. Each call gets the field
paths and the document, and returns a value or `null` for each.

`null` is the expected answer for most of them. A DMP does not usually state a ROR
identifier, a byte size or a licence URL, and a wrong value is worse than none.
'''),

code("ask", '''
llm = OllamaModel(model=MODEL, host="http://localhost:11434", num_ctx=32768)

# One group per top-level dmp key, with the plain scalars bundled together.
CHUNK = 15   # a call asking for more than this starts dropping fields

subtrees = {}
for path in ALL_PATHS:
    parts = path.split(".")
    key = parts[1].replace("[]", "") if len(parts) > 1 else "dmp"
    name = key if len(parts) > 2 or "[]" in path else "basics"
    subtrees.setdefault(name, []).append(path)

groups = {}
for name, paths in subtrees.items():
    if len(paths) <= CHUNK:
        groups[name] = paths
    else:
        for i in range(0, len(paths), CHUNK):
            groups[f"{name} {i // CHUNK + 1}"] = paths[i:i + CHUNK]

print(f"{len(ALL_PATHS)} fields in {len(groups)} calls\\n")

SYSTEM = """You read Data Management Plans and report what they say.

If the document does not state something, answer null. Do not guess and do not
invent identifiers, emails, ORCIDs, DOIs, grant numbers or dates. Most fields will
be null - that is the correct answer, not a failure."""


def ask(paths, text):
    """Ask for one group of field paths; returns {path: value or None}."""
    lines = ["Read the Data Management Plan below and report these fields.", ""]
    lines += [f"  {p}" for p in paths]
    lines += ["", "Answer null for anything the document does not state.", "",
              "--- DOCUMENT ---", text, "--- END DOCUMENT ---"]
    fmt = {"type": "object",
           "properties": {p: {"type": ["string", "null"]} for p in paths},
           "required": paths}
    raw = llm.complete(SYSTEM, "\\n".join(lines), schema=fmt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\\{.*\\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


values = {}
for name, paths in groups.items():
    print(f"  {name:22} {len(paths):>3} fields ...", flush=True)
    values.update(ask(paths, clean))

print(f"\\ngot answers for {len(values)} fields")
'''),

md("s5", '''
## Step 5 — Fill the skeleton

Values are written into the skeleton at their own paths, so everything lands where
the standard puts it. Anything the model answered `null` for keeps the skeleton's
`null`.
'''),

code("fill", '''
def is_empty(value):
    return value is None or str(value).strip().lower() in ("", "null", "none", "n/a")


def set_path(obj, path, value):
    """Write value into obj at a dotted path; '[]' means the first list entry."""
    parts = path.split(".")
    cur = obj
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        is_list = part.endswith("[]")
        key = part[:-2] if is_list else part
        if last:
            if is_list:
                if not isinstance(cur.get(key), list) or not cur[key]:
                    cur[key] = [None]
                cur[key][0] = value
            else:
                cur[key] = value
            return
        if is_list:
            if not isinstance(cur.get(key), list) or not cur[key]:
                cur[key] = [{}]
            cur = cur[key][0]
        else:
            if not isinstance(cur.get(key), dict):
                cur[key] = {}
            cur = cur[key]


result = json.loads(json.dumps(SKELETON))   # fresh copy, all null

filled = []
for path in ALL_PATHS:
    value = values.get(path)
    if not is_empty(value):
        set_path(result, path, value)
        filled.append(path)

print(f"filled {len(filled)} of {len(ALL_PATHS)} fields; "
      f"{len(ALL_PATHS) - len(filled)} left null\\n")
for path in filled:
    print(f"  {path:52} {str(values[path])[:48]}")
'''),

md("s6", '''
## Step 6 — Check it against the schema

Validated against the schema as published, `required` lists and all. Anything
reported here is a field the standard wants that this document did not supply —
information about the DMP, not a bug.
'''),

code("validate", '''
import jsonschema

errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(result),
                key=lambda e: list(e.path))

if not errors:
    print("Valid against maDMP-schema-1.2.")
else:
    print(f"{len(errors)} schema complaint(s) - mostly nulls where the standard "
          f"wants a value:\\n")
    for e in errors[:25]:
        where = "/".join(str(p) for p in e.path) or "(root)"
        print(f"  {where:38} {e.message[:78]}")
    if len(errors) > 25:
        print(f"  ... and {len(errors) - 25} more")
'''),

md("s7", '''
## Step 7 — Save it
'''),

code("save", '''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"saved -> {OUT}  ({OUT.stat().st_size:,} bytes)\\n")
print(json.dumps(result, indent=2, ensure_ascii=False)[:2500])
'''),

md("check", '''
---

### Quick sanity check

Emails, ORCIDs and DOIs are what a model invents most readily, and an invented one
looks exactly like a real one. This just asks whether each is in the PDF.
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
    print(f"  {value:46} {'in the PDF' if value.lower() in low else 'NOT IN THE PDF'}")
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
