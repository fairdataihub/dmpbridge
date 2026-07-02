# dmpbridge

Extract and label the structure of Data Management Plan (DMP) PDF documents using local or cloud LLMs.

- **Batch inference** — extract text with pdfplumber, classify each block with an LLM in sliding windows, output labeled JSON.
- **Whole-document inference** — send the entire document in a single LLM call for higher-context classification.
- **PDF-direct inference** — send the raw PDF to a vision-capable model (Claude); no pdfplumber, paragraph-level output.
- **Evaluation** — compare predictions against manually labeled ground truth with per-label F1, confusion matrices, and error breakdowns.
- **Viewer** — side-by-side PDF + JSON browser UI with bounding-box overlays.

---

## Labels

Each text block is classified as one of five labels:

| Label | Description |
|-------|-------------|
| `title` | The single main title of the document |
| `section.title` | A numbered section heading |
| `section.description` | Funder template text describing what the section must cover |
| `question.text` | A sub-question or sub-topic prompt within a section |
| `answer.text` | The researcher's actual written response |

---

## Model performance

Evaluated against 10 manually labeled DMP samples:

| Model | Provider | Strategy | Block acc | Gold acc |
|-------|----------|----------|-----------|----------|
| `claude-opus-4-8` | Anthropic | Whole-doc | **96.9%** | 98.9% |
| `claude-opus-4-8` | Anthropic | PDF-direct | 95.1% | **99.5%** |
| `claude-opus-4-8` | Anthropic | Batch | 94.9% | 96.8% |
| `llama3.3:70b` | Ollama (local) | Batch | 94.1% | 96.0% |
| `llama3.3:70b` | Ollama (local) | Whole-doc | 91.8% | 93.7% |
| `llama3.1:8b` | Ollama (local) | Whole-doc | 84.2% | 86.0% |
| `llama3.1:8b` | Ollama (local) | Batch | 67.3% | 68.7% |

> **Block acc** = correct / all predicted blocks.  
> **Gold acc** = correct / all manually labeled gold items — comparable across strategies regardless of block count.

---

## Project structure

```
dmpbridge/
├── __init__.py
│
├── core/                        # Pipeline, config, and output conversion
│   ├── config.py                # Model, host, API keys, batch size
│   ├── pipeline.py              # process_pdf() — extract → classify → convert
│   └── converter.py             # Flat labeled blocks → nested DMP Tool JSON
│
├── strategies/                  # Classification strategies
│   ├── batch.py                 # BatchStrategy — sliding-window LLM calls
│   ├── wholedoc.py              # WholeDocStrategy — single full-document call
│   └── pdf_direct.py            # PdfDirectStrategy — raw PDF to vision model
│
├── models/                      # LLM backend implementations
│   ├── ollama.py                # Ollama (local)
│   ├── anthropic.py             # Anthropic Claude
│   ├── openai.py                # OpenAI
│   └── gemini.py                # Google Gemini
│
├── preprocess/                  # PDF text extraction
│   ├── pdfplumber_reader.py     # Line-level block extraction
│   ├── text_cleaner.py          # Remove duplicate words from extracted text
│   └── page_images.py           # Per-page PNG images with bounding-box overlays
│
├── parsers/                     # LLM response parsing
│   └── json_parser.py           # Robust JSON extraction from LLM output
│
├── prompts/                     # Prompt content
│   ├── labels.py                # Label definitions and output schema
│   └── system.py                # System prompt with few-shot examples
│
├── evaluation/                  # Evaluation and experiment management
│   ├── evaluate.py              # Metrics, confusion matrix, F1 — dmpbridge-evaluate CLI
│   └── experiment.py            # Config-driven experiment runner — dmpbridge-experiment CLI
│
├── cli/                         # Command-line entry points
│   ├── main.py                  # dmpbridge — batch inference CLI
│   └── wholedoc_cmd.py          # dmpbridge-wholedoc — whole-doc inference CLI
│
└── utils/                       # Shared utilities
    ├── logger.py                # Console + rotating file logging
    └── exceptions.py            # Custom exception hierarchy

data/
├── input/
│   ├── pdfs/                    # Source DMP PDFs (sample1.pdf … sample10.pdf)
│   └── ground_truth/            # Hand-labeled JSON files (sampleN_dmp.json)
│       └── reference_text/      # Plain-text reference extracts
│
└── output/
    ├── extracted/               # Raw pdfplumber blocks before labeling
    └── labeled/                 # LLM output, organized by experiment
        ├── claude-opus-4-8_batch/        # sampleN.json, sampleN_structured.json
        ├── claude-opus-4-8_whole_doc/
        ├── claude-opus-4-8_pdf/
        ├── llama3.3-70b_batch/
        └── …

experiments/                     # YAML experiment configs
├── claude-opus-4-8-batch.yaml
├── claude-opus-4-8-wholedoc.yaml
└── …

notebooks/
├── eval/                        # Per-model evaluation notebooks
│   ├── claude-opus-4-8.ipynb
│   ├── llama3.3-70b.ipynb
│   └── llama3.1-8b.ipynb
├── experiments/                 # Strategy comparison and experiment logs
└── exploration/                 # Early exploration notebooks

logs/                            # Rotating log files (excluded from git)
templates/
└── index.html                   # Viewer UI
main.py                          # FastAPI server for the browser viewer
dmpbridge.html                   # Standalone viewer (no server needed)
```

---

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install the package

```powershell
pip install -e .
```

For notebook support (pandas, matplotlib, seaborn):

```powershell
pip install -e ".[notebooks]"
```

For the browser viewer (FastAPI server):

```powershell
pip install -e ".[server]"
```

### 3. Configure your provider

Edit **[dmpbridge/core/config.py](dmpbridge/core/config.py)** or create a `.env` file in the project root:

```env
# Ollama (local)
DMPBRIDGE_PROVIDER=ollama
DMPBRIDGE_MODEL=llama3.3:70b
DMPBRIDGE_HOST=http://localhost:11434

# Anthropic (Claude)
DMPBRIDGE_PROVIDER=anthropic
DMPBRIDGE_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
DMPBRIDGE_PROVIDER=openai
DMPBRIDGE_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Google Gemini
DMPBRIDGE_PROVIDER=gemini
DMPBRIDGE_MODEL=gemini-2.0-flash
GEMINI_API_KEY=...
```

For Ollama, install from **https://ollama.com** and pull a model:

```powershell
ollama pull llama3.3:70b     # best local accuracy (~42 GB)
ollama pull llama3.1:8b      # faster, less memory (~5 GB)
```

---

## Usage

### Batch inference — `dmpbridge`

```powershell
# Run with default provider and model (from config / .env)
dmpbridge document.pdf

# Override provider and model
dmpbridge document.pdf --provider anthropic --model claude-opus-4-8

# Save to a specific path, skip raw extraction save
dmpbridge document.pdf -o output.json --no-raw

# Show detailed progress logs
dmpbridge document.pdf -v
```

### Whole-document inference — `dmpbridge-wholedoc`

```powershell
# Ollama
dmpbridge-wholedoc --provider ollama --model llama3.3:70b

# Anthropic
dmpbridge-wholedoc --provider anthropic --model claude-opus-4-8

# Custom sample range
dmpbridge-wholedoc --start 3 --end 6
```

### PDF-direct inference — `dmpbridge-pdf`

Sends the raw PDF to a vision-capable model (Claude). No pdfplumber — Claude extracts and classifies in one call.

```powershell
dmpbridge-pdf --model claude-opus-4-8
```

### Config-driven experiments — `dmpbridge-experiment`

```powershell
# Run an experiment defined by a YAML config
dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml

# Run and immediately evaluate
dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml --evaluate

# List all available configs
dmpbridge-experiment --list
```

### Evaluation — `dmpbridge-evaluate`

```powershell
# Evaluate all samples for the default model (from config)
dmpbridge-evaluate

# Evaluate a single output file
dmpbridge-evaluate data/output/labeled/claude-opus-4-8_batch/sample1.json
```

### Benchmark — compare all experiments

```powershell
python experiments/benchmark.py
```

### Python API

```python
from dmpbridge import process_pdf

# Batch inference (default)
blocks = process_pdf("document.pdf", output="labeled.json")

# Use a specific strategy
from dmpbridge.strategies import get_strategy

strategy = get_strategy("wholedoc", provider="anthropic", model="claude-opus-4-8")
blocks = process_pdf("document.pdf", strategy=strategy)

# Convert to nested DMP Tool schema
from dmpbridge import to_structured
structured = to_structured(blocks)
```

### Viewer (PDF + JSON side by side)

**Standalone** — open `dmpbridge.html` in any browser and drag-drop a PDF + labeled JSON.

**Server** — start the FastAPI server and open `http://localhost:8000`:

```powershell
uvicorn main:app --reload
```

---

## Adding a new experiment

1. Create a YAML config in `experiments/`:

```yaml
name: My experiment
strategy: batch          # batch | wholedoc | pdf_direct
model: llama3.3:70b
provider: ollama
batch_size: 10
context_size: 3
pdf_dir: data/input/pdfs
out_dir: data/output/labeled
sample_start: 1
sample_end: 10
```

2. Run it:

```powershell
dmpbridge-experiment experiments/my-experiment.yaml --evaluate
```

Output is written to `data/output/labeled/{model}_{strategy}/sampleN.json`.
