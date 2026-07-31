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

