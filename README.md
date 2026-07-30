# DMPBridge

**DMPBridge** is a research pipeline for extracting and structuring the content of Data Management Plan (DMP) PDF documents using local large language models via [Ollama](https://ollama.com).

> **Status:** Active research project — methods, results, and structure are evolving.

---

## What it does

DMP documents mix funder-written instructions with researcher-written responses inside the same PDF. DMPBridge reads those PDFs and classifies each text block into one of five semantic labels, producing structured JSON that downstream tools can process.

The pipeline supports two inference strategies, a full evaluation framework comparing predictions against manually labeled ground truth, and a leave-2-out cross-validation design for measuring prompt robustness.

---

## Labels

Each text block is classified as one of:

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

Pull a model (choose based on your hardware):

```bash
ollama pull llama3.3:70b       # ~42 GB — best accuracy
ollama pull llama3.3:8b        # ~5 GB  — fast, good quality
ollama pull llama3.1:8b        # ~5 GB  — baseline 8B
ollama pull gemma4:4b          # ~2.5 GB — smallest / fastest

```

Configure in `.env` at the project root:

```env
DMPBRIDGE_PROVIDER=ollama
DMPBRIDGE_MODEL=llama3.3:70b
DMPBRIDGE_HOST=http://localhost:11434
DMPBRIDGE_BATCH_SIZE=10
```

---

## Basic usage

```bash
# Label a single PDF (batch strategy, default model)
dmpbridge document.pdf

# Specify model
dmpbridge document.pdf --model llama3.3:8b

# Whole-document strategy (single model call, better for small models)
dmpbridge-wholedoc --model llama3.1:8b
```

---

## Inference strategies

| Strategy | How it works | Best for |
|---|---|---|
| `batch` | Blocks sent in sliding windows of 10 with 3-block context overlap | Large models (70B+) |
| `wholedoc` | All blocks sent in a single model call | Small models (≤8B) |


---

## Running experiments

Each experiment is defined by a YAML config file in `experiments/`.

```bash
# Run a full experiment (all 10 samples)
dmpbridge-experiment experiments/llama3.3-70b-batch.yaml

# Evaluate results against ground truth
dmpbridge-evaluate

# List available experiments
dmpbridge-experiment --list
```

### Available experiments

| ID | Model | Strategy | Config file |
|---|---|---|---|
| M03 | Llama 3.3 70B | batch | `llama3.3-70b-batch.yaml` |
| M04 | Llama 3.3 70B | whole-doc | `llama3.3-70b-wholedoc.yaml` |
| M05 | Llama 3.1 8B | batch | `llama3.1-8b-batch.yaml` |
| M06 | Llama 3.1 8B | whole-doc | `llama3.1-8b-wholedoc.yaml` |
| M07 | Llama 3.3 8B | batch | `llama3.3-8b-batch.yaml` |
| M08 | Llama 3.3 8B | whole-doc | `llama3.3-8b-wholedoc.yaml` |
| M09 | Gemma 4 4B | batch | `gemma4-4b-batch.yaml` |
| M10 | Gemma 4 4B | whole-doc | `gemma4-4b-wholedoc.yaml` |

---

## Evaluation

The evaluation framework compares predicted labels against manually annotated ground truth in `data/input/ground_truth/`.

**Metrics computed:**
- **Block accuracy** — fraction of predicted blocks with the correct label
- **Per-label F1** — precision, recall, F1 for each of the 5 labels
- **Confusion matrix** — raw counts + recall-normalised heatmap with a `missed` column for gold items the extractor never surfaced
- **Confidence calibration** — every predicted block carries a `confidence` score (0.0–1.0); calibration plots show whether stated confidence tracks actual accuracy

### Confidence scores

The LLM assigns a confidence score to every block alongside its label. The five tiers are:

| Score | Meaning |
|---|---|
| 1.0 | Unambiguous — only one label is consistent |
| 0.8–0.9 | High — one label is clearly best |
| 0.6–0.7 | Moderate — two labels are plausible |
| 0.4–0.5 | Low — evidence is roughly balanced |
| < 0.4 | Very uncertain |

Blocks below the review threshold (default 0.75) are flagged for human review.

### Leave-2-out cross-validation

The rotation experiment design uses 2 gold DMPs as few-shot examples, evaluates on the remaining 8, and rotates through 5 pairs. Low variance across rotations means the prompt generalises rather than overfitting to specific examples.

```bash
python experiments/run_rotations.py
```

---

## Project layout

```
dmpbridge/                  Python package
  strategies/               Inference strategies (batch, wholedoc)
  models/                   Ollama model backend
  prompts/                  System prompt, label schema, few-shot builder
  evaluation/               Metrics, confusion matrix, confidence calibration
  core/                     Config, pipeline, converter
  cli/                      CLI entry points
  parsers/                  LLM JSON response parser
  preprocess/               PDF extraction (pdfplumber)

experiments/                YAML experiment configs
  rotations/                Leave-2-out rotation configs (r1–r5)

data/
  input/
    pdfs/                   Sample PDFs (sample1.pdf … sample10.pdf)
    ground_truth/           Manual annotations (sampleN_dmp.json)
  output/
    labeled/                LLM-labeled JSON (one sub-dir per experiment tag)
    page_images/            Page PNGs for vision strategy

notebooks/
  000_experiment_log.ipynb        Live accuracy table for all experiments
  001_pdfplumber_extraction.ipynb PDF extraction quality check
  002_pdfplumber_extraction_eval.ipynb Extraction vs ground truth alignment
  003_strategy_comparison.ipynb   Cross-model, cross-strategy comparison
  004_llama3.1-8b.ipynb           Llama 3.1 8B deep-dive
  005_llama3.3-70b.ipynb          Llama 3.3 70B deep-dive
  006_llama3.3-8b.ipynb           Llama 3.3 8B deep-dive
  007_gemma4-4b.ipynb             Gemma 4 4B deep-dive
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

strategy = get_strategy("batch", model="llama3.3:70b")
blocks   = strategy.run(Path("document.pdf"))

# Blocks with confidence scores
for b in blocks:
    print(b["label"], b["confidence"], b["text"][:60])
```

```python
from dmpbridge.evaluation.evaluate import load_method, compute_f1_rows

df, confusion, errors = load_method("llama3.3-70b_batch")
print(df)                          # per-sample accuracy
print(compute_f1_rows(confusion))  # per-label F1
```

---

## Current results

| Model | Strategy | Accuracy |
|---|---|---|
| Llama 3.3 70B | batch | 93.8% |
| Llama 3.3 70B | whole-doc | 91.8% |
| Llama 3.1 8B | batch | 67.2% |
| Llama 3.1 8B | whole-doc | 83.3% |
| Llama 3.3 8B | batch | Pending |
| Gemma 4 4B | batch | Pending |

Evaluated on 10 manually labeled DMP documents, 741 blocks total.  
Detailed per-label F1, confusion matrices, and confidence calibration: `notebooks/003_strategy_comparison.ipynb`.
