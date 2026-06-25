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

## Project structure

```
dmpbridge/
├── __init__.py     # exports process_pdf, to_structured, convert_file
├── extractor.py    # pdfplumber text extraction + page image export
├── classifier.py   # Ollama LLM classifier (few-shot + context window)
├── pipeline.py     # combines extraction + classification
├── converter.py    # converts flat labeled JSON → hierarchical manual schema
├── cli.py          # dmpbridge command-line tool
└── config.py       # ← edit here to change model / host / batch size

data/
├── pdfsamples/     # sample DMP PDFs
├── manuallabeled/  # hand-labeled ground truth JSON  (<sample>_dmp.json)
├── llmlabeled/     # LLM output: flat (<sample>_<model>.json)
│                   #             structured (<sample>_<model>_structured.json)
└── pdfplumber/     # (auto-generated) raw pdfplumber JSON before labeling

notebooks/
├── 01_pdfplumber_batch_test.ipynb     # batch extraction across all sample PDFs
├── 02_evaluation_pdfplumber_batch_test.ipynb
├── 03_eval_llama3.3-70b.ipynb         # evaluation: confusion matrix + F1 charts (llama3.3:70b)
├── 03_eval_llama3.1-8b.ipynb          # evaluation: confusion matrix + F1 charts (llama3.1:8b)
└── 03_comparison_dashboard.ipynb      # side-by-side model comparison + error analysis

templates/
└── index.html      # Viewer UI served by FastAPI

evaluate.py         # CLI evaluation script (confusion matrix + per-label F1)
main.py             # FastAPI server
dmpbridge.html      # Standalone viewer (no server needed)
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
ollama pull llama3.3:70b     # best accuracy (requires ~42 GB)
ollama pull llama3.1:8b      # faster, less memory (~5 GB)
```

Set your chosen model in `dmpbridge/config.py`.

---

## Pipeline (PDF → labeled JSON)

### Configure the model

Edit **[dmpbridge/config.py](dmpbridge/config.py)**:

```python
MODEL      = "llama3.3:70b"   # any model installed in Ollama
HOST       = "http://localhost:11434"
BATCH_SIZE = 10
```

### CLI

```powershell
# Run pipeline
dmpbridge document.pdf

# Specify output path
dmpbridge document.pdf -o data/llmlabeled/output.json

# Also produce hierarchical structured JSON
dmpbridge document.pdf -o data/llmlabeled/sample1_llama3.3-70b.json --structured

# Override model for this run
dmpbridge document.pdf --model llama3.1:8b -o data/llmlabeled/sample1_llama3.1-8b.json

# Show detailed progress
dmpbridge document.pdf -v

# Skip saving the raw pdfplumber JSON
dmpbridge document.pdf --no-raw
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

For interactive charts (confusion matrix, F1 scores, model comparison):

```powershell
jupyter lab notebooks/03_eval_llama3.3-70b.ipynb   # single-model deep dive
jupyter lab notebooks/03_comparison_dashboard.ipynb  # cross-model comparison
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
