"""Build notebooks/pdf-to-rda-dmp-json.ipynb.

One prompt: the pdfplumber text of a DMP plus the complete, unmodified maDMP 1.2
schema, with strict instructions to follow the schema. Run for every sample in
SAMPLES with llama3.1:8b, gemma4:e4b and llama3.3:70b, and save every result.

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
strict instructions to follow the schema — first with `llama3.1:8b`, then the same
prompt with `gemma4:e4b` and `llama3.3:70b`.

Every run of the notebook is saved in its own folder, `data/output/rda/runs/v1/`,
`v2/`, `v3/` ... — the result JSON per model, the exact prompt (`prompt.txt`) and the
timings (`run.json`). So you can change the prompt a little, run all cells again, and
nothing from an earlier run is lost. The run number is picked automatically; set
`RUN_NAME` in the settings if you want a name of your own.

| Step | What happens |
|---|---|
| 1 | Read sample 14 with pdfplumber |
| 2 | Load the schema — unchanged |
| 3 | Prompt → `llama3.1:8b` |
| 4 | Same prompt → `gemma4:e4b` |
| 5 | Same prompt → `llama3.3:70b` |
| 6 | Save this run — results, prompt and timings under `runs/<RUN_NAME>/` |
'''),

code("setup", '''
import json
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.models.ollama import OllamaModel

SAMPLES  = [14]
PDF_DIR  = Path("data/input/pdfs")
SCHEMA   = Path("data/output/rda/maDMP-schema-1.2.json")
HOST     = "http://localhost:11434"
NUM_CTX  = 32768
MODEL_1  = "llama3.1:8b"
MODEL_2  = "gemma4:e4b"
MODEL_3  = "llama3.3:70b"
OUT_DIR  = Path("data/output/rda")

# Every run is saved in its own folder, data/output/rda/runs/<RUN_NAME>/, with its prompt.
# Leave RUN_NAME = None and each run gets the next number (v1, v2, v3 ...) by itself;
# or give it a name of your own. An existing run is never overwritten.
RUN_NAME = None

import re
saved = sorted(p.name for p in (OUT_DIR / "runs").glob("*") if p.is_dir())
if RUN_NAME is None:
    numbers = [int(s[1:]) for s in saved if re.fullmatch(r"v\\d+", s)]
    RUN_NAME = f"v{1 + max(numbers, default=0)}"
elif RUN_NAME in saved:
    raise SystemExit(f"Run {RUN_NAME!r} already exists. Set RUN_NAME = None for the next free number, "
                     f"or pick a name not in: {', '.join(saved)}")
RUN_DIR = OUT_DIR / "runs" / RUN_NAME
print(f"this run: {RUN_NAME}   (earlier runs: {', '.join(saved) or 'none'})")
'''),

md("s1", '''
## Step 1 — Read the PDFs
'''),

code("read", '''
dmp_texts = {}
for n in SAMPLES:
    dmp_texts[n] = get_extractor("pdfplumber").extract(PDF_DIR / f"sample{n}.pdf")[0]["text"]
    print(f"sample{n}: {len(dmp_texts[n]):,} characters")
    print(dmp_texts[n][:300])
    print()
'''),

md("s2", '''
## Step 2 — Load the schema

The schema is used exactly as published — nothing removed, nothing changed.
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

Before each call the model is loaded and checked: if it does not sit **100% on
the GPU** it is skipped, because a model spilling onto the CPU turns minutes into
hours.
'''),

code("prompt", '''
import subprocess
import time

import requests

SYSTEM = """You convert Data Management Plans into RDA maDMP JSON.

Strict rules:
1. Use only the field names defined in the schema. Never add a key that is not in the schema.
2. Put every field exactly where the schema places it. The whole document is one top-level "dmp" object.
3. Where the schema lists allowed values, use one of them, spelled exactly as in the schema .
4. Take every value from the Data Management Plan text. Never copy example values from the schema and never hallucinate.
5. Output only the JSON object. No explanation, no markdown."""


def make_prompt(dmp_text):
    return f"""Here is the RDA maDMP JSON schema (version 1.2). Follow it strictly:

{schema_text}

Here is the text of a Data Management Plan:

{dmp_text}

Generate one JSON object that strictly follows the schema above, filling in
whatever information the Data Management Plan text contains. Output only the JSON."""


def run(model, sample):
    """Send one sample's prompt to one model; the schema is also the output
    constraint. Prints the time taken and the tokens in and out."""
    for other in (MODEL_1, MODEL_2, MODEL_3):
        if other != model:
            subprocess.run(["ollama", "stop", other], check=False)   # one model in VRAM at a time

    # Load the model, then make sure all of it is on the GPU before the real call.
    t0 = time.perf_counter()
    requests.post(f"{HOST}/api/generate", timeout=1800,
                  json={"model": model, "keep_alive": -1, "options": {"num_ctx": NUM_CTX}})
    load = time.perf_counter() - t0
    loaded = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    placement = next((l for l in loaded.splitlines() if l.startswith(model)), "")
    if "100% GPU" not in placement:
        print(f"{model}, sample{sample}: SKIPPED - the model is not fully on the GPU")
        print(f"ollama ps says: {' '.join(placement.split()) or 'not loaded'}")
        print()
        subprocess.run(["ollama", "stop", model], check=False)
        return None

    llm = OllamaModel(model=model, host=HOST, num_ctx=NUM_CTX, num_predict=8000)
    t0 = time.perf_counter()
    result = json.loads(llm.complete(SYSTEM, make_prompt(dmp_texts[sample]), schema=schema_full))
    elapsed = time.perf_counter() - t0

    s = llm.last_call
    timings[model, sample] = {"seconds": round(elapsed), "model_load_seconds": round(load),
                              "tokens_sent": s["prompt_eval_count"], "tokens_generated": s["eval_count"]}
    print(f"{model}, sample{sample}: {elapsed:.0f} s   (loading the model took {load:.0f} s)")
    print(f"tokens sent to the model: {s['prompt_eval_count']:,}   tokens generated: {s['eval_count']:,}")
    print()
    return result


def run_all(model):
    """One model over every sample; prints each result."""
    for n in SAMPLES:
        results[model, n] = run(model, n)
        if results[model, n] is None:
            for rest in SAMPLES:                 # skipped once: don't reload it per sample
                results.setdefault((model, rest), None)
            break
        print(json.dumps(results[model, n], indent=2, ensure_ascii=False))
        print()


results, timings = {}, {}
run_all(MODEL_1)
'''),

md("s4", '''
## Step 4 — Same prompt → gemma4:e4b
'''),

code("gemma", '''
run_all(MODEL_2)
'''),

md("s5", '''
## Step 5 — Same prompt → llama3.3:70b

The 70B needs about 53 GB of VRAM at this context size — all three 24 GB cards.
With fewer cards available it cannot sit fully on the GPU and is skipped.
'''),

code("llama33", '''
run_all(MODEL_3)
'''),

md("s6", '''
## Step 6 — Save this run

Everything goes into `data/output/rda/runs/<RUN_NAME>/`: the result JSON per model,
`prompt.txt` (the exact prompt that produced them) and `run.json` (timings and tokens).
The results are also copied to `data/output/rda/` as the current ones.

**Next:** open `notebooks/evaluate-rda-json.ipynb`, set `RUN` to this run's name
(printed below), run it.
'''),

code("save", r'''
import shutil
from datetime import datetime

RUN_DIR.mkdir(parents=True, exist_ok=True)
for (model, n), result in results.items():
    if result is None:
        print(f"{model:14} sample{n}: skipped, nothing saved")
        continue
    name = f"sample{n}.rda.{model.replace(':', '-')}.json"
    (RUN_DIR / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(RUN_DIR / name, OUT_DIR / name)                  # also the "current" copy
    print(f"{model:14} sample{n} -> {RUN_DIR / name}")

# The exact prompt and settings behind these results, so the run can be reproduced.
(RUN_DIR / "prompt.txt").write_text(
    "SYSTEM:\n" + SYSTEM + "\n\nPROMPT (the schema and the document text go in the braces):\n"
    + make_prompt("{dmp_text}").replace(schema_text, "{schema_text}"), encoding="utf-8")
(RUN_DIR / "run.json").write_text(json.dumps({
    "run": RUN_NAME, "date": datetime.now().isoformat(timespec="minutes"), "samples": SAMPLES,
    "models": [MODEL_1, MODEL_2, MODEL_3], "schema": SCHEMA.name, "num_ctx": NUM_CTX,
    "timings": {f"{m} sample{n}": t for (m, n), t in timings.items()},
    "skipped": [f"{m} sample{n}" for (m, n), r in results.items() if r is None],
}, indent=2), encoding="utf-8")
print(f"\nprompt.txt and run.json -> {RUN_DIR}")
print(f"\nNext: in notebooks/evaluate-rda-json.ipynb set RUN = {RUN_NAME!r} and run it.")
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
