# Configuration

[← Main README](../README.md) · [All docs](README.md)

Every setting the pipeline accepts, in one place. For just running it, the
[README quick start](../README.md#quick-start) is enough — this page is the reference you
come back to.

## The short version

Two settings cover almost everything:

```bash
dmpbridge my-plan.pdf --model gemma4:e4b
```

- **`--model`** — which Ollama model labels the text. The built-in default is
  `llama3.3:70b` (~40 GB), so pass `gemma4:e4b` (~3 GB) unless you have the big one. Set it
  once with `DMPBRIDGE_MODEL` and you can drop the flag entirely.
- **`--extractor` / `--fallback`** — how the PDF is read. Leave both alone: the defaults
  read a normal PDF and OCR a broken one.

For the RDA JSON as well, give each part its own model:

```bash
dmpbridge-pipeline --narrative-model gemma4:e4b --rda-model llama3.1:8b --start 1 --end 1
```

The rest of this page is the full reference.

| Looking for | Section |
|---|---|
| where PDFs and rules live | [Inputs](#inputs) |
| the YAML config file | [Settings in YAML](#settings-in-yaml--democonfigyaml) |
| every command-line flag | [Settings on the command line](#settings-on-the-command-line) |
| calling it from Python | [From Python](#from-python) |
| `DMPBRIDGE_*` variables | [Environment variables](#environment-variables) |

---

## Inputs

| What | Where | Notes |
|---|---|---|
| DMP PDFs | `data/input/pdfs/` | named `sample1.pdf`, `sample2.pdf`, … — the `--start`/`--end` range indexes these |
| Annotation rules | `data/input/Rules.xlsx` | drives stage 4; **column order is load-bearing**, see [scoring.md](scoring.md) |
| Reference annotations | `data/input/ground_truth_old_version/`, `ground_truth_new_version/` | only needed for scoring |
| RDA reference annotations | `data/input/rda_ground_truth/` | only needed for scoring the RDA JSON |

To run a PDF that isn't part of the sample set, use the single-document command — it takes
any path and doesn't need the `sampleN.pdf` naming:

```bash
dmpbridge my-plan.pdf --model gemma4:e4b
```

---

## Settings in YAML — `demo/config.yaml`

One YAML file describes a run. This is what `scripts/run_demo.py` and
`notebooks/demo-from-yaml-config.ipynb` read:

```yaml
name: Demo run
strategy: wholedoc
provider: ollama
host: http://localhost:11434

narrative:                  # labels the text, builds the DMP Tool JSON (stages 2-4)
  model: gemma4:e4b         # any model already pulled in Ollama
rda:                        # fills in the RDA JSON, then the merge (stages 5-6)
  model: llama3.1:8b        # leave the rda: block out for the narrative part only

extractor: pdfplumber       # pdfplumber | lightonocr | docling
fallback: auto              # auto = LightOnOCR | docling | null to disable

pdf_dir: data/input/pdfs    # expects sample1.pdf, sample2.pdf, ...
sample_start: 1
sample_end: 1               # keep this small for a quick demo run
```

`extractor` is what reads a normal PDF; `fallback` is what rescues one whose text layer is
broken. They are separate settings because the rescue is conditional — the fallback only
runs on a document that fails the readability check, so clean PDFs are untouched.

A plain `model: gemma4:e4b` in place of the `narrative:` block also works.

The experiment files in `experiments/`, read by `dmpbridge-experiment`, run the narrative
part only and take `model:` / `models:` rather than the two blocks.
`experiments/full-example.yaml` shows every field they accept, including multiple models,
multiple extractors, and the `evaluation:` list that drives scoring.

---

## Settings on the command line

### `dmpbridge-pipeline` — the sample set, all six stages

Runs `dmpbridge-wholedoc` for stages 1–4, then the RDA part for stages 5–6.

| Flag | Default | What it does |
|---|---|---|
| `--narrative-model` | *(required)* | model for stages 2–4 |
| `--rda-model` | *(required)* | model for stage 5, the RDA JSON |
| `--extractor` | `pdfplumber` | `pdfplumber`, `lightonocr` or `docling` |
| `--fallback` | `auto` | `auto` re-reads a document with LightOnOCR when its text extracts as garbage |
| `--start` / `--end` | `1` / `10` | inclusive sample range |
| `--pdf-dir` | `data/input/pdfs` | where the PDFs live |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--no-evaluate` | off | skip scoring the RDA JSON against the hand-made references |

### `dmpbridge-evaluate-rda` — score the RDA JSON

| Argument | Default | What it does |
|---|---|---|
| `<rda-tag>` | — | a folder in `data/output/5_rda/`, e.g. `llama3.1-8b_pdfplumber_rda` |
| `--start` / `--end` | every sample | inclusive sample range |
| `--list` | off | list the RDA runs and exit |
| `--no-save` | off | print only; write no figure or spreadsheet |

See [Scoring → The RDA JSON](scoring.md#the-rda-json) for what the numbers mean.

### `dmpbridge-rda` — stages 5–6 only

For when the narrative part has already run. Same flags as `dmpbridge-pipeline`, except
the models:

| Flag | Default | What it does |
|---|---|---|
| `--model` | *(required)* | model for the RDA JSON |
| `--narrative-model` | same as `--model` | whose `4_final` narrative JSON to merge with |

### `dmpbridge-wholedoc` — the sample set, all four stages

| Flag | Default | What it does |
|---|---|---|
| `--model` | `llama3.3:70b` | any model pulled in Ollama |
| `--extractor` | `pdfplumber` | `pdfplumber`, `lightonocr` or `docling` |
| `--fallback` | off | `auto` re-reads a document with LightOnOCR when its text extracts as garbage |
| `--start` / `--end` | `1` / `10` | inclusive sample range |
| `--pdf-dir` | `data/input/pdfs` | where the PDFs live |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--no-cache` | off | re-extract even when stage 1 already has the document |
| `--no-rules` | off | stop after stage 3, skip the rule-converted final JSON |
| `--no-save-native` | off | skip the extractor's raw native dump |
| `--force-ocr` | off | docling only — OCR every page instead of trusting the text layer |

### `dmpbridge` — one PDF, any path

| Flag | Default | What it does |
|---|---|---|
| `--model` | `llama3.3:70b` | any model pulled in Ollama |
| `--extractor` | `pdfplumber` | `pdfplumber`, `lightonocr` or `docling` |
| `--fallback` | `auto` | `auto` = LightOnOCR; `off` disables the rescue |
| `-o` / `--output` | `<pdf>_labeled.json` | where the labeled blocks go |
| `--structured` | `<output>_structured.json` | where the DMP Tool JSON goes |
| `--no-structured` | off | skip the structured output |
| `--no-raw` | off | skip saving the raw extraction JSON |
| `--save-images` | off | also write per-page PNGs with block boxes |
| `-v` / `--verbose` | off | detailed progress logs |

**The default model is `llama3.3:70b`**, which is around 40 GB. Pass `--model gemma4:e4b`
(or set `DMPBRIDGE_MODEL`) unless you have actually pulled the 70B.

---

## From Python

For embedding DMPBridge in an application rather than calling it from a shell.
`process_pdf` takes the same options as the `dmpbridge` command:

```python
import dmpbridge

blocks = dmpbridge.process_pdf(
    "my-plan.pdf",
    model="gemma4:e4b",
    extractor="pdfplumber",
    fallback="auto",              # or a name, or an ordered list of names
    structured_output="structured.json",
)
```

To run the whole sample set from Python, drive it from a config object instead — this is
what `scripts/run_demo.py` and the demo notebook do:

```python
from dmpbridge.evaluation.experiment import Experiment
from dmpbridge.rda.config import load_pipeline_config

cfg, rda_model = load_pipeline_config("demo/config.yaml")   # rda_model is None without an rda: block
Experiment(cfg).run()                                        # stages 1-4
```

`scripts/run_demo.py` then runs stages 5–6 with `dmpbridge.rda.stages.run_rda_stages`. To
get the RDA JSON for a piece of text directly:

```python
from dmpbridge.rda import extract_rda

rda, info = extract_rda(plan_text, model="llama3.1:8b")    # rda == {"dmp": {...}}
```

---

## Environment variables

| Variable | Overrides |
|---|---|
| `DMPBRIDGE_MODEL` | the default model |
| `DMPBRIDGE_PROVIDER` | the provider (only `ollama` is supported) |
| `DMPBRIDGE_HOST` | the Ollama URL |

Ollama itself needs several environment variables on a multi-GPU machine — see
[troubleshooting.md](troubleshooting.md).
