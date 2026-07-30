# DMPBridge

**DMPBridge** is a research pipeline for extracting and structuring the content of Data Management Plan (DMP) PDF documents using local large language models via [Ollama](https://ollama.com).

> **Status:** Active research project — methods, results, and structure are evolving.

---

## What it does

DMP documents mix funder-written instructions with researcher-written responses inside the same PDF. DMPBridge reads those PDFs and classifies each text block into one of five semantic labels, producing structured JSON that downstream tools can process.

The pipeline supports three PDF extraction backends (**pdfplumber**, **Docling**, **LightOnOCR**) and any number of Ollama models. A single YAML config defines which models × extractors to run, and the system produces one result set per combination.

---

## Labels

| Label | What it represents |
|---|---|
| `title` | The main document title |
| `section.title` | A section heading |
| `section.description` | Funder-written instructions for the section |
| `question.text` | A sub-question or sub-topic prompt |
| `answer.text` | The researcher's written response |

---

## Setup

**Requirements:** Python 3.10+ · [Ollama](https://ollama.com) running locally

```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate     # macOS / Linux

pip install -e .
```

Pull the models you want to use:

```bash
ollama pull llama3.3:70b       # ~42 GB — best accuracy
ollama pull llama3.1:8b        # ~5 GB  — fast, good quality
ollama pull gemma4:e4b         # ~3 GB  — efficient small model
```

### Optional extraction backends

By default only **pdfplumber** is installed (included in the base install).  
Install additional backends as needed:

```bash
pip install -e ".[docling]"         # Docling (IBM ML layout analyser)
pip install -e ".[lighton]"         # LightOnOCR-2-1B (HuggingFace transformer)
pip install -e ".[all-extractors]"  # Both at once
```

Configure defaults in `.env` at the project root (optional):

```env
DMPBRIDGE_PROVIDER=ollama
DMPBRIDGE_MODEL=llama3.3:70b
DMPBRIDGE_HOST=http://localhost:11434
```

---

## Running experiments

### Experiment YAML format

Every experiment is a YAML file that defines which models and extractors to run.  
`models` and `extractors` are lists — the system runs every combination automatically.

```yaml
# experiments/llama3.1-8b-wholedoc.yaml
name: Llama 3.1 8B — Whole-doc
strategy: wholedoc
models: [llama3.1:8b]
extractors: [pdfplumber, docling, lighton]
provider: ollama
host: http://localhost:11434
prompt: default
pdf_dir: data/input/pdfs
out_dir: data/output/labeled
sample_start: 1
sample_end: 10
```

The above produces three output directories:

```
data/output/labeled/
  llama3.1-8b_whole_doc/          ← pdfplumber results
  llama3.1-8b_docling_whole_doc/  ← Docling results
  llama3.1-8b_lighton_whole_doc/  ← LightOnOCR results
```

### Run a single-model experiment

```bash
# Llama 3.1 8B — all three extractors
dmpbridge-experiment experiments/llama3.1-8b-wholedoc.yaml

# Gemma 4 E4B — all three extractors
dmpbridge-experiment experiments/gemma4-e4b-wholedoc.yaml

# Llama 3.3 70B — all three extractors (requires ~42 GB VRAM, slow)
dmpbridge-experiment experiments/llama3.3-70b-wholedoc.yaml
```

### Run the full matrix (all models × all extractors)

```bash
dmpbridge-experiment experiments/wholedoc.yaml
```

This runs 3 models × 3 extractors = **9 combinations** over 10 PDFs.

### Available experiment configs

| Config | Models | Extractors |
|---|---|---|
| `experiments/wholedoc.yaml` | llama3.1:8b, gemma4:e4b, llama3.3:70b | pdfplumber, docling, lighton |
| `experiments/llama3.1-8b-wholedoc.yaml` | llama3.1:8b | pdfplumber, docling, lighton |
| `experiments/gemma4-e4b-wholedoc.yaml` | gemma4:e4b | pdfplumber, docling, lighton |
| `experiments/llama3.3-70b-wholedoc.yaml` | llama3.3:70b | pdfplumber, docling, lighton |

---

## Quick single-run CLI

For a fast one-off inference without a YAML file:

```bash
# Default model (from .env or llama3.3:70b), pdfplumber extractor
dmpbridge-wholedoc

# Choose model and extractor explicitly
dmpbridge-wholedoc --model llama3.1:8b --extractor pdfplumber
dmpbridge-wholedoc --model gemma4:e4b   --extractor docling
dmpbridge-wholedoc --model llama3.1:8b  --extractor lighton

# Limit to a subset of samples
dmpbridge-wholedoc --model llama3.1:8b --start 3 --end 6

# All options
dmpbridge-wholedoc --help
```

Output goes to `data/output/labeled/{model-slug}_whole_doc/` (pdfplumber)  
or `data/output/labeled/{model-slug}_{extractor}_whole_doc/` (docling / lighton).

---

## Evaluation

Compare predicted labels against manually annotated ground truth:

```bash
dmpbridge-evaluate
```

**Metrics computed:**
- **Block accuracy** — fraction of predicted blocks with the correct label
- **Per-label F1** — precision, recall, F1 for each of the 5 labels
- **Confusion matrix** — raw counts + recall-normalised heatmap with a `missed` column for gold items the extractor never surfaced
- **Confidence calibration** — every predicted block carries a `confidence` score (0.0–1.0); calibration plots show whether stated confidence tracks actual accuracy

### Confidence scores

| Score | Meaning |
|---|---|
| 1.0 | Unambiguous — only one label is consistent |
| 0.8–0.9 | High — one label is clearly best |
| 0.6–0.7 | Moderate — two labels are plausible |
| 0.4–0.5 | Low — evidence is roughly balanced |
| < 0.4 | Very uncertain |

Blocks below the review threshold (default 0.75) are flagged for human review.

---

## Leave-2-out cross-validation

The rotation design uses 2 gold DMPs as few-shot examples, evaluates on the remaining 8, and rotates through 5 pairs. Results are reported per `(model, extractor)` combination.

```bash
# Run all 5 rotations — all models × all extractors
python experiments/run_rotations.py

# Skip inference, re-evaluate existing results only
python experiments/run_rotations.py --evaluate-only

# Filter to one model or one extractor
python experiments/run_rotations.py --model llama3.1:8b
python experiments/run_rotations.py --extractor pdfplumber
python experiments/run_rotations.py --model gemma4:e4b --extractor docling
```

Output shows a table per rotation, then a summary:

```
  Model             Extractor     Mean gold acc   Std dev
  ──────────────────────────────────────────────────────
  gemma4:e4b        pdfplumber         90.8%      3.1%
  llama3.1:8b       pdfplumber         70.2%      5.4%
  ...
```

---

## Extraction backends

| Backend | Install | Bbox/font data | Speed | Notes |
|---|---|---|---|---|
| `pdfplumber` | base install | full | fast | default; best for clean, text-layer PDFs |
| `docling` | `.[docling]` | bbox only (no font) | medium | IBM ML layout analyser; good for complex layouts |
| `lighton` | `.[lighton]` | none (OCR only) | slow | HuggingFace `lightonai/LightOnOCR-2-1B`; handles scanned PDFs |

Blocks from Docling and LightOnOCR carry `None` for unavailable fields (`avg_font_size`, `font_names`, bbox for LightOnOCR). The visualisation tools skip those fields silently.

---

## Project layout

```
dmpbridge/                  Python package
  strategies/               Inference strategy (wholedoc)
  extractors/               PDF extraction backends
    base.py                 BaseExtractor ABC
    pdfplumber_extractor.py pdfplumber wrapper
    docling_extractor.py    Docling wrapper
    lighton_extractor.py    LightOnOCR-2-1B wrapper
  models/                   Ollama model backend
  prompts/                  System prompt, label schema, few-shot builder
  evaluation/               Metrics, confusion matrix, confidence calibration
  core/                     Config, pipeline, converter
  cli/                      CLI entry points
  parsers/                  LLM JSON response parser
  preprocess/               pdfplumber text utilities

experiments/                YAML experiment configs
  wholedoc.yaml             Master config — all models × all extractors
  llama3.1-8b-wholedoc.yaml
  gemma4-e4b-wholedoc.yaml
  llama3.3-70b-wholedoc.yaml
  rotations/                Leave-2-out rotation configs (r1–r5, 3 models each)
  run_rotations.py          Rotation runner + summary reporter
  benchmark.py              Cross-experiment benchmark table

data/
  input/
    pdfs/                   Sample PDFs (sample1.pdf … sample10.pdf)
    ground_truth/           Manual annotations (sampleN_dmp.json)
  output/
    labeled/                LLM-labeled JSON (one sub-dir per experiment tag)

notebooks/
  000_experiment_log.ipynb        Live accuracy table for all experiments
  001_pdfplumber_extraction.ipynb PDF extraction quality check
  002_pdfplumber_extraction_eval.ipynb Extraction vs ground truth alignment
  003_strategy_comparison.ipynb   Cross-model comparison
  004_llama3.1-8b.ipynb           Llama 3.1 8B deep-dive
  005_llama3.3-70b.ipynb          Llama 3.3 70B deep-dive
  007_gemma4-e4b.ipynb            Gemma 4 E4B deep-dive
```

---

## Output format

Each labeled JSON file contains a list of blocks:

```json
[
  {
    "text": "Data Management Plan",
    "label": "title",
    "confidence": 0.97,
    "page": 1,
    "is_bold": true
  },
  ...
]
```

A companion `_structured.json` file reorganises the same blocks into the DMP Tool narrative schema (title → sections → questions → answers).

---

## Python API

```python
from dmpbridge.strategies import get_strategy
from pathlib import Path

# pdfplumber (default)
strategy = get_strategy("wholedoc", model="llama3.1:8b")
blocks   = strategy.run(Path("document.pdf"))

# Docling extractor
strategy = get_strategy("wholedoc", model="gemma4:e4b", extractor="docling")
blocks   = strategy.run(Path("document.pdf"))

for b in blocks:
    print(b["label"], b["confidence"], b["text"][:60])
```

```python
from dmpbridge.evaluation.evaluate import load_method, compute_f1_rows

df, confusion, errors = load_method("llama3.1-8b_whole_doc")
print(df)                          # per-sample accuracy
print(compute_f1_rows(confusion))  # per-label F1
```

---

## Current results

| Model | Extractor | Accuracy |
|---|---|---|
| Gemma 4 E4B | pdfplumber | 90.8% |
| Llama 3.1 8B | pdfplumber | 70.2% |
| Llama 3.3 70B | pdfplumber | Pending |
| All models | docling / lighton | Pending |

Evaluated on 10 manually labeled DMP documents.  
Detailed per-label F1, confusion matrices, and confidence calibration: `notebooks/003_strategy_comparison.ipynb`.
