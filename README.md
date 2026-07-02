# DMPBridge

**DMPBridge** is a research pipeline for extracting and structuring the content of Data Management Plan (DMP) PDF documents using large language models.

> **Status:** Active research project — methods, results, and structure are evolving.

---

## What it does

DMP documents mix funder-written instructions with researcher-written responses inside the same PDF. DMPBridge reads those PDFs and classifies each block of text into one of five semantic labels, producing structured JSON that downstream tools can process.

The pipeline supports multiple LLM providers (local and cloud) and several inference strategies. An evaluation framework compares predictions against manually labeled ground truth.

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

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

pip install -e .
```

Configure your provider in a `.env` file at the project root:

```env
DMPBRIDGE_PROVIDER=ollama          # ollama | anthropic | openai | gemini
DMPBRIDGE_MODEL=llama3.3:70b
ANTHROPIC_API_KEY=...              # if using Anthropic
```

---

## Basic usage

```bash
# Label a single PDF
dmpbridge document.pdf

# Run a full experiment from a config file
dmpbridge-experiment experiments/my-experiment.yaml

# Evaluate results against ground truth
dmpbridge-evaluate
```

---

## Project layout

```
dmpbridge/      Python package (strategies, models, evaluation, CLI)
experiments/    YAML configs — one file per experiment
data/           Input PDFs, ground truth annotations, pipeline outputs
notebooks/      Analysis and evaluation notebooks
report-doc/     Project reports
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) for local inference, or API keys for cloud providers
