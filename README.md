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
| `section.description` | Instructions for the section |
| `question.text` | A question or prompt |
| `answer.text` | The researcher's written response |

---

## How the pipeline works (big picture)

```
PDF file
   |
   v
1. Extraction (pdfplumber / Docling / LightOnOCR)
   |     Turns the PDF into text blocks, line by line, with position,
   |     font size, and bold/italic flags.
   v
2. LLM classification (Ollama)
   |     One model call labels every block: title / section.title /
   |     section.description / question.text / answer.text.
   v
3. Structured JSON (dmpbridge/core/converter.py)
   |     Labeled blocks are nested into the DMP Tool schema:
   |     narrative -> template -> section[] -> question[] -> answer.
   v
4. Evaluation — two independent scoring paths, same structured JSON
   |
   +-- Path A: score directly against data/input/ground_truth_old_version/
   |
   +-- Path B: apply the annotation-conversion rule (fills in empty
       question.text fields from the section/document title) to get
       "final JSON", then score that against
       data/input/ground_truth_new_version/
```

Path A and B share the same matching/scoring engine underneath (greedy gold-oriented matching, confusion matrix, per-label F1, micro-averaged Precision/Recall/F1) — they only differ in which ground truth, and whether the annotation-conversion rule is applied first.

**Status:** Path A is the original, stable evaluation. Path B (the new-annotation rule) is validated against 10 hand-checked samples but has one known gap — it doesn't merge multiple short sub-question labels within a section, a pattern seen in one sample so far that isn't reliably derivable from the data yet. See `notebooks/annotation_conversion_test.ipynb` for the full derivation and what's still open.

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

## Running the pipeline (as of now)

### One PDF

```bash
dmpbridge path/to/document.pdf --model llama3.3:70b
```

Writes `document_labeled.json` (flat blocks) and `document_labeled_structured.json` (nested DMP Tool schema) next to the input by default.

### Batch — `sample1.pdf` … `sample10.pdf`

```bash
dmpbridge-wholedoc --model llama3.1:8b --extractor pdfplumber
dmpbridge-wholedoc --model llama3.3:70b --start 3 --end 10   # subset
```

Writes to `data/output/labeled/<model>_<extractor>_whole_doc/`. Safe to re-run — samples that already have output are skipped.

### Batch — reproducible, YAML-driven

```bash
dmpbridge-experiment experiments/llama3.3-70b-wholedoc.yaml
dmpbridge-experiment experiments/llama3.3-70b-wholedoc.yaml --evaluate   # + Path A summary after running
dmpbridge-experiment --list                                              # see available configs
```

Runs every model × extractor combination defined in the config in one call.

---

## Evaluation — two paths, same structured JSON

### Path A — score against the original ground truth

```bash
dmpbridge-evaluate --list                                   # see available result tags
dmpbridge-evaluate llama3.1-8b_pdfplumber_whole_doc          # one tag, all samples
dmpbridge-evaluate llama3.1-8b_pdfplumber_whole_doc --exclude 1,2   # skip prompt-dev samples
dmpbridge-evaluate path/to/sampleN_structured.json            # single file
```

### Path B — apply the annotation rule, score against the new ground truth

```bash
dmpbridge-evaluate-new --list
dmpbridge-evaluate-new llama3.1-8b_pdfplumber_whole_doc       # converts to "final JSON", then scores
dmpbridge-evaluate-new llama3.1-8b_pdfplumber_whole_doc --convert-only   # just build final JSON
```

Both print a confusion matrix, per-label precision/recall/F1, and an overall micro-averaged Precision/Recall/F1 (the single number that penalizes both missed items *and* over-generated/spurious ones — see `dmpbridge.evaluation.evaluate.micro_prf1`).

---

## Notebooks

| Notebook | What it's for |
|---|---|
| `model_comparison_pdfplumber.ipynb` | Presentation-ready comparison of all models, both Path A and Path B, side by side |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Covers the structured-JSON converter state machine, the evaluation matching/scoring engine, and the shared batch-runner helper.

