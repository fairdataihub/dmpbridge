# Configuration

Every setting the pipeline accepts, in one place. For just running it, the
[README quick start](../README.md#quick-start) is enough — this page is the reference you
come back to.

---

## Inputs

| What | Where | Notes |
|---|---|---|
| DMP PDFs | `data/input/pdfs/` | named `sample1.pdf`, `sample2.pdf`, … — the `--start`/`--end` range indexes these |
| Annotation rules | `data/input/Rules.xlsx` | drives stage 4; **column order is load-bearing**, see [scoring.md](scoring.md) |
| Reference annotations | `data/input/ground_truth_old_version/`, `ground_truth_new_version/` | only needed for scoring |

To run a PDF that isn't part of the sample set, use the single-document command — it takes
any path and doesn't need the `sampleN.pdf` naming:

```bash
dmpbridge my-plan.pdf --model gemma4:e4b
```

---

## Settings in YAML — `demo/config.yaml`

One YAML file describes a run. This is what `scripts/run_demo.py`,
`notebooks/demo-from-yaml-config.ipynb` and `dmpbridge-experiment` all read:

```yaml
name: Demo run
strategy: wholedoc
provider: ollama
host: http://localhost:11434

model: llama3.1:8b          # any model already pulled in Ollama

extractor: pdfplumber       # pdfplumber | lightonocr | docling
fallback: auto              # auto = LightOnOCR | docling | null to disable

pdf_dir: data/input/pdfs    # expects sample1.pdf, sample2.pdf, ...
sample_start: 1
sample_end: 1               # keep this small for a quick demo run
```

`extractor` is what reads a normal PDF; `fallback` is what rescues one whose text layer is
broken. They are separate settings because the rescue is conditional — the fallback only
runs on a document that fails the readability check, so clean PDFs are untouched.

`experiments/full-example.yaml` shows every field the format accepts, including multiple
models, multiple extractors, and the `evaluation:` list that drives scoring.

---

## Settings on the command line

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

## Environment variables

| Variable | Overrides |
|---|---|
| `DMPBRIDGE_MODEL` | the default model |
| `DMPBRIDGE_PROVIDER` | the provider (only `ollama` is supported) |
| `DMPBRIDGE_HOST` | the Ollama URL |

Ollama itself needs several environment variables on a multi-GPU machine — see
[troubleshooting.md](troubleshooting.md).
