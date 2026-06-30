# dmpbridge

Extract and label the structure of Data Management Plan (DMP) PDF documents.

1. **Pipeline** — extract text from a PDF with pdfplumber, classify each block using a local LLM (via Ollama), and output a labeled JSON file.
2. **Viewer** — a side-by-side PDF + JSON browser UI that overlays bounding boxes on the PDF, synchronized with the JSON table.
3. **Evaluation** — compare LLM predictions against manually labeled ground truth with a confusion matrix and per-label F1 scores.

---

## Labels

Each text block is classified as one of:

| Label | Description |
|-------|-------------|
| `title` | The single main title of the document |
| `section.title` | A numbered section heading (e.g. "1. Data sharing and preservation") |
| `section.description` | Funder template text describing what the section must cover |
| `question.text` | A sub-question or sub-topic prompt within a section |
| `answer.text` | The researcher's actual written response |

---

## How it works

### Step 1 — Extract text blocks from the PDF

pdfplumber reads every line with full character-level detail (font name, size, bounding box). Each line becomes a **block dict** with fields: `text`, `page`, `x0/top/x1/bottom`, `avg_font_size`, `is_bold`, `is_italic`, `label` (initially `null`).

### Step 2 — Save the raw extraction to disk

Before the LLM touches anything, the raw block list is saved as JSON (default: `data/pdfplumber/<name>.json`). This lets you inspect or debug the pdfplumber output independently of the classification. Skip with `--no-raw`.

### Step 3 — Classify blocks with the LLM

Blocks are sent to a local Ollama LLM in small batches (default: 10 blocks per request). Each batch includes the last 3 already-labeled blocks as context so the model can track where it is in the document. The model receives a system prompt with:

- Definitions of all 5 labels
- Key decision rules (e.g. "should/must" phrasing → `section.description`; researcher narrative → `answer.text`)
- Real few-shot examples from actual DMP documents
- A JSON schema that forces the output format

Temperature is set to 0 for deterministic output.

### Step 4 — Save the flat labeled JSON

The complete block list — one entry per line, with the `label` field now filled in — is saved to the output path (default: `<pdf_name>_labeled.json`).

### Step 5 — Convert to the nested DMP Tool schema

The flat list is converted to the nested DMP Tool JSON schema by a positional state machine:

- `section.title` opens a new section
- `section.description` before any question → goes into the section's description field
- `section.description` after a question has started → stays in document reading order (continues the question text or answer, whichever is open)
- Consecutive `question.text` blocks with no answer yet → merged into one question
- `answer.text` → appended to the current question's answer

The converter trusts the LLM's labels exactly — it does not relabel or reinterpret content.

### Step 6 — Save the structured JSON

The nested JSON is saved alongside the flat file (default: `<pdf_name>_labeled_structured.json`). Skip with `--no-structured`.

---

## Model performance

Evaluated against 10 manually labeled DMP samples:

| Model | Size | Accuracy | Notes |
|-------|------|----------|-------|
| `llama3.3:70b` | ~42 GB | ~85–90% | Best accuracy; recommended for production |
| `llama3.1:8b` | ~5 GB | ~67% | Fast, runs on laptop GPU; good for testing |

---

## Project structure

```
dmpbridge/
├── __init__.py     # exports process_pdf, to_structured, convert_file
├── extractor.py    # pdfplumber text extraction + page image export
├── classifier.py   # Ollama LLM classifier (few-shot + sliding context window)
├── pipeline.py     # combines extraction + classification + conversion
├── converter.py    # converts flat labeled blocks → nested DMP Tool schema
├── cli.py          # dmpbridge command-line tool
└── config.py       # ← edit here to change model / host / batch size

data/
├── pdfsamples/     # sample DMP PDFs
├── manuallabeled/  # hand-labeled ground truth JSON  (<sample>_dmp.json)
├── llmlabeled/     # LLM output: flat (<sample>_<model>.json)
│                   #             structured (<sample>_<model>_structured.json)
└── pdfplumber/     # (auto-generated) raw pdfplumber JSON before labeling

notebooks/
├── 01_pdfplumber_batch_test.ipynb         # batch extraction across all sample PDFs
├── 02_evaluation_pdfplumber_batch_test.ipynb
├── 03_eval_llama3.3-70b.ipynb             # confusion matrix + F1 charts (llama3.3:70b)
├── 03_eval_llama3.1-8b.ipynb             # confusion matrix + F1 charts (llama3.1:8b)
└── 03_comparison_dashboard.ipynb          # side-by-side model comparison + error analysis

templates/
└── index.html      # Viewer UI served by FastAPI

evaluate.py                   # CLI evaluation script (confusion matrix + per-label F1)
main.py                       # FastAPI server for the browser viewer
dmpbridge.html                # Standalone viewer (no server needed)
DMP_Labeling_Strategy.docx    # Strategy overview document (prompt design, models, evaluation)
pyproject.toml
requirements.txt
```

---

## Setup

### 1. Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

### 3. Install Ollama

Download from **https://ollama.com** and install, then pull a model:

```powershell
ollama pull llama3.3:70b     # best accuracy (requires ~42 GB RAM/VRAM)
ollama pull llama3.1:8b      # faster, less memory (~5 GB)
```

Set your chosen model in `dmpbridge/config.py`.

---

## Pipeline (PDF → labeled JSON)

### Configure the model

Edit **[dmpbridge/config.py](dmpbridge/config.py)**:

```python
MODEL      = "llama3.3:70b"          # any model installed in Ollama
HOST       = "http://localhost:11434"
BATCH_SIZE = 10                       # blocks per LLM request
```

### CLI

```powershell
# Basic run — produces <pdf>_labeled.json and <pdf>_labeled_structured.json
dmpbridge document.pdf

# Specify output path
dmpbridge document.pdf -o data/llmlabeled/output.json

# Override model for this run
dmpbridge document.pdf --model llama3.1:8b -o data/llmlabeled/sample1_llama3.1-8b.json

# Show detailed progress (logs each batch)
dmpbridge document.pdf -v

# Skip saving the raw pdfplumber JSON
dmpbridge document.pdf --no-raw

# Skip writing the structured JSON (flat labeled JSON only)
dmpbridge document.pdf --no-structured

# Save per-page PNG images with colored bounding boxes per label
dmpbridge document.pdf --save-images data/pdfplumber
```

### Python API

```python
from dmpbridge import process_pdf

blocks = process_pdf("document.pdf", output="labeled.json")

# Both flat + structured in one call
blocks = process_pdf(
    "document.pdf",
    output="labeled.json",
    structured_output="labeled_structured.json",
)
```

### Convert an existing flat file to structured

```python
from dmpbridge import convert_file

convert_file("data/llmlabeled/sample1_llama3.3-70b.json")
```

The structured JSON follows the DMP Tool narrative schema. `id` fields (`template.id`, `section.id`, `question.id`, `answer.id`) are omitted because they cannot be determined from a PDF — they can be added downstream once the record is stored in the DMP Tool database.

---

## Evaluation

Compare LLM output against manually labeled ground truth in `data/manuallabeled/`.

```powershell
# Evaluate all samples
python evaluate.py

# Evaluate a single file
python evaluate.py data/llmlabeled/sample1_llama3.3-70b.json
```

The evaluation uses **token containment** to match predicted blocks to gold items: a predicted block matches a gold entry if 75% or more of the block's words appear in the gold text. Both a forward pass (predicted → gold) and a reverse pass (gold → predicted, to find missed items) are run.

Output includes:
- Per-sample accuracy table
- Confusion matrix (true label vs predicted label)
- Per-label precision, recall, and F1 score
- List of gold items the LLM produced no matching block for

For interactive charts (confusion matrix, F1 scores, model comparison):

```powershell
jupyter lab notebooks/03_eval_llama3.3-70b.ipynb    # single-model deep dive
jupyter lab notebooks/03_comparison_dashboard.ipynb   # cross-model comparison
```

---

## Viewer (PDF + JSON side by side)

### Option A — Standalone HTML (no server needed)

Open `dmpbridge.html` in any modern browser. Drag and drop a PDF and its labeled JSON onto the page.

### Option B — FastAPI server

```powershell
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

Open **http://localhost:8000** — upload files through the browser UI.
