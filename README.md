# dmpbridge

Extract and label the structure of Data Management Plan (DMP) PDF documents using local or cloud LLMs.

1. **Pipeline** — extract text from a PDF with pdfplumber, classify each block with an LLM, and output a labeled JSON file.
2. **Whole-document inference** — send the entire document to the model in a single call for higher-context classification.
3. **Evaluation** — compare LLM predictions against manually labeled ground truth with a confusion matrix and per-label F1 scores.
4. **Viewer** — a side-by-side PDF + JSON browser UI that overlays bounding boxes on the PDF, synchronized with the JSON table.

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

### Batch inference (default)

Blocks are sent to the LLM in small batches (default: 10 blocks per request). Each batch includes the last 3 already-labeled blocks as context so the model can track where it is in the document. Temperature is set to 0 for deterministic output.

### Whole-document inference

All blocks in the document are sent in a single request. The model sees the full document at once — better for structural continuity, but slower and more expensive for long documents.

### Conversion

The flat labeled block list is converted to the nested DMP Tool JSON schema by a positional state machine that groups blocks into sections, questions, and answers in document reading order.

---

## Model performance

Evaluated against 10 manually labeled DMP samples:

| Model | Provider | Strategy | Accuracy |
|-------|----------|----------|----------|
| `claude-opus-4-8` | Anthropic | Whole-doc | **96.9%** |
| `claude-opus-4-8` | Anthropic | Batch | 94.9% |
| `llama3.3:70b` | Ollama (local) | Batch | 94.1% |
| `llama3.3:70b` | Ollama (local) | Whole-doc | 91.8% |
| `llama3.1:8b` | Ollama (local) | Whole-doc | 84.2% |
| `llama3.1:8b` | Ollama (local) | Batch | 67.3% |

---

## Project structure

```
dmpbridge/
├── __init__.py         # exports process_pdf, to_structured, convert_file
├── config.py           # model / host / API keys / batch size — edit here
├── prompt.py           # all prompt content: labels, system prompt, prompt builders
├── extractor.py        # pdfplumber text extraction + page image export
├── classifier.py       # batch LLM classifier (Ollama, OpenAI, Anthropic, Gemini)
├── wholedoc.py         # whole-document inference helpers
├── runner.py           # whole-document CLI (dmpbridge-wholedoc)
├── pipeline.py         # combines extraction + classification + conversion
├── converter.py        # flat labeled blocks → nested DMP Tool schema
├── evaluate.py         # evaluation logic + notebook helpers (dmpbridge-evaluate)
├── cli.py              # dmpbridge command-line tool
├── logging_setup.py    # shared logging config (console + rotating file)
└── exceptions.py       # custom exception hierarchy

data/
├── pdfsamples/         # sample DMP PDFs
├── manuallabeled/      # hand-labeled ground truth JSON  (<sample>_dmp.json)
├── llmlabeled/         # LLM output: flat and structured labeled JSON
└── pdfplumber/         # raw pdfplumber extraction before labeling

notebooks/
└── eval/
    ├── llama3.3-70b.ipynb       # confusion matrix + F1 charts (llama3.3:70b)
    ├── llama3.1-8b.ipynb        # confusion matrix + F1 charts (llama3.1:8b)
    └── claude-opus-4-8.ipynb    # confusion matrix + F1 charts (Claude)

logs/                   # rotating log file (excluded from git)
templates/
└── index.html          # Viewer UI served by FastAPI
main.py                 # FastAPI server for the browser viewer
dmpbridge.html          # Standalone viewer (no server needed)
pyproject.toml
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

### 3. Configure providers

Edit **[dmpbridge/config.py](dmpbridge/config.py)** or set environment variables in a `.env` file:

```env
# For Ollama (local)
PROVIDER   = ollama
MODEL      = llama3.3:70b
HOST       = http://localhost:11434

# For Anthropic (Claude)
PROVIDER          = anthropic
MODEL             = claude-opus-4-8
ANTHROPIC_API_KEY = sk-ant-...
```

For Ollama, install from **https://ollama.com** and pull a model:

```powershell
ollama pull llama3.3:70b     # best local accuracy (~42 GB)
ollama pull llama3.1:8b      # faster, less memory (~5 GB)
```

---

## Usage

### Batch inference (PDF → labeled JSON)

```powershell
# Basic run
dmpbridge document.pdf

# Override model or provider for this run
dmpbridge document.pdf --provider anthropic --model claude-opus-4-8

# Show detailed progress
dmpbridge document.pdf -v
```

### Whole-document inference

```powershell
# Ollama
dmpbridge-wholedoc --provider ollama --model llama3.3:70b

# Anthropic
dmpbridge-wholedoc --provider anthropic --model claude-opus-4-8
```

### Evaluation

```powershell
# Evaluate all samples (uses default model from config)
dmpbridge-evaluate

# Evaluate a single file
dmpbridge-evaluate data/llmlabeled/sample1_llama3.3-70b_batch.json
```

For interactive charts — confusion matrices, F1 comparisons, and error breakdowns:

```powershell
jupyter lab notebooks/eval/llama3.3-70b.ipynb
```

### Python API

```python
from dmpbridge import process_pdf

blocks = process_pdf("document.pdf", output="labeled.json")
```

### Viewer (PDF + JSON side by side)

**Standalone** — open `dmpbridge.html` in any browser and drag-drop a PDF + labeled JSON.

**Server** — `uvicorn main:app --reload` then open `http://localhost:8000`.
