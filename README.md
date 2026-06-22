# dmpbridge

Extract and label the structure of Data Management Plan (DMP) PDF documents.

1. **Pipeline** — extract text from a PDF with pdfplumber, classify each block using a local LLM (via Ollama), and output a labeled JSON file.
2. **Viewer** — a side-by-side PDF + JSON browser UI that overlays bounding boxes on the PDF, synchronized with the JSON table.

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
├── __init__.py     # exports process_pdf
├── extractor.py    # pdfplumber text extraction + page image export
├── classifier.py   # Ollama LLM classifier (few-shot + context window)
├── pipeline.py     # combines extraction + classification
├── cli.py          # dmpbridge command-line tool
└── config.py       # ← edit here to change model / host / batch size

data/
├── pdfsamples/     # sample DMP PDFs
├── manuallabeled/  # hand-labeled ground truth JSON
├── llmlabeled/     # LLM-labeled JSON output
└── pdfplumber/     # (auto-generated) raw pdfplumber JSON before labeling

notebooks/
├── 01_pdfplumber_batch_test.ipynb
└── 02_evaluation_pdfplumber_batch_test.ipynb

templates/
└── index.html      # Viewer UI served by FastAPI

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

### How it works

```
PDF
 ↓ pdfplumber
Line-level text blocks  (page, coordinates, font size, bold, italic)
 ↓ saved to data/pdfplumber/<name>.json  (raw, before labeling)
 ↓ Ollama LLM — batched in groups of 10
   · Few-shot examples from manually labeled DMPs guide each label
   · Last 3 labeled blocks are sent as context for each new batch
Labeled blocks: title | section.title | section.description | question.text | answer.text
 ↓ saved to labeled JSON
```

### Configure the model

Edit **[dmpbridge/config.py](dmpbridge/config.py)**:

```python
MODEL      = "llama3.3:70b"   # any model installed in Ollama
HOST       = "http://localhost:11434"
BATCH_SIZE = 10
```

### CLI

```powershell
# Run pipeline — saves raw JSON to data/pdfplumber/, labeled JSON next to the PDF
dmpbridge document.pdf

# Specify output path
dmpbridge document.pdf -o data/llmlabeled/output.json

# Override model for this run
dmpbridge document.pdf --model llama3.1:8b

# Show detailed progress
dmpbridge document.pdf -v

# Skip saving the raw pdfplumber JSON
dmpbridge document.pdf --no-raw
```

### Python API

```python
from dmpbridge import process_pdf

blocks = process_pdf("document.pdf", output="labeled.json")

# Override model
blocks = process_pdf("document.pdf", model="llama3.1:8b", output="labeled.json")

# Inspect label counts
from collections import Counter
print(Counter(b["label"] for b in blocks))
# Counter({'answer.text': 52, 'question.text': 18, 'section.description': 8,
#          'section.title': 4, 'title': 1})
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

---

## End-to-end workflow

```
1. Run the pipeline
   dmpbridge data/pdfsamples/sample1.pdf -o data/llmlabeled/sample1_labeled.json

2. Open the viewer
   Open dmpbridge.html in a browser  (or run: uvicorn main:app --reload)

3. Load files
   Load data/pdfsamples/sample1.pdf + data/llmlabeled/sample1_labeled.json

4. Inspect
   Click any row in the table → the corresponding block highlights in the PDF
   Use the Label filter to show only sections, questions, or answers
```
