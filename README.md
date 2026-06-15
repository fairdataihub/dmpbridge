# dmpbridge

A tool for extracting and analyzing the structure of Data Management Plan (DMP) PDF documents.

1. **Pipeline** — extract text from a PDF with pdfplumber, classify each block as `document_title`, `section`, `subsection`, or `content` using a local LLM (via Ollama), and output a labeled JSON file.
2. **Evaluation** — compare extracted text against reference files using nine quality metrics (word capture, ROUGE-L, precision, recall, F1, and more).
3. **Viewer** — a side-by-side PDF + JSON browser UI that overlays bounding boxes on the PDF, synchronized with the JSON table.

---

## Project structure

```
dmpbridge/
├── __init__.py        # exports process_pdf
├── extractor.py       # pdfplumber text extraction + page image export
├── classifier.py      # Ollama LLM classifier
├── pipeline.py        # combines extraction + classification
├── cli.py             # dmpbridge command-line tool
├── config.py          # ← edit here to change model / host / batch size
└── evaluation/
    ├── __init__.py
    └── pdfplumber_text_evaluator.py  # text quality metrics

notebooks/
├── 01_pdfplumber_batch_test.ipynb       # batch extraction across all sample PDFs
└── 02_evaluation_pdfplumber_batch_test.ipynb  # evaluation metrics against reference text

data/
├── pdfsamples/        # sample DMP PDFs (10 documents)
├── reference_text/    # manually curated reference text files for evaluation
├── llmlabeled/        # LLM-generated labeled JSON output
├── manuallabeled/     # manually corrected labeled JSON
├── pdfplumber/        # (auto-generated) raw pdfplumber JSON, one file per PDF
├── pdfplumber_extracted_blocks/       # line-level JSON blocks from batch notebook
├── pdfplumber_extracted_text/         # plain .txt extraction per PDF
├── pdfplumber_extracted_markdown/     # .md extraction per PDF
└── pdfplumber_extracted_blocks_debug/ # debug CSVs + batch summary

templates/
└── index.html         # Viewer UI served by FastAPI

main.py                # FastAPI server
dmpbridge.html         # Standalone viewer (no server needed)
pyproject.toml         # package install config
requirements.txt       # FastAPI dependencies
venv/                  # virtual environment (not in git)
```

---

## Setup

### 1. Create and activate the virtual environment

```powershell
# Create (one time)
python -m venv venv

# Activate (every session)
.\venv\Scripts\Activate.ps1
```

### 2. Install everything

```powershell
pip install -r requirements.txt
pip install -e .
```

### 3. Install Ollama (for LLM labeling)

Download from **https://ollama.com** and install.

Pull a model — any of these work:

```powershell
ollama pull llama3.2:latest      # 2 GB — fast, good for testing
ollama pull llama3.1:8b          # 4.7 GB — more accurate
ollama pull llama3.3:8b          # newest llama3 variant
```

---

## Part 1 — Pipeline (PDF → labeled JSON)

### How it works

```
PDF
 ↓ pdfplumber
Line-level text blocks (page, coordinates, font size, bold flag)
 ↓ save raw JSON  →  data/pdfplumber/<name>.json
 ↓ Ollama LLM (batched, structured output)
Labeled blocks  →  document_title | section | subsection | content
 ↓ save labeled JSON
```

### Configure the model

Open **[dmpbridge/config.py](dmpbridge/config.py)** and set your preferred model:

```python
MODEL = "llama3.1:8b"

HOST       = "http://localhost:11434"   # Ollama server URL
BATCH_SIZE = 10                         # blocks per LLM request
```

### CLI usage

```powershell
# Basic — raw pdfplumber JSON auto-saved to data/pdfplumber/, labeled JSON next to the PDF
dmpbridge document.pdf

# Specify labeled output path
dmpbridge document.pdf -o data/llmlabeled/output.json

# Override model for this run (ignores config.py)
dmpbridge document.pdf --model llama3.1:8b

# Show detailed progress per batch
dmpbridge document.pdf -v

# Save raw pdfplumber JSON to a custom folder instead of the default
dmpbridge document.pdf --raw-dir my/raw/folder

# Skip saving the raw pdfplumber JSON
dmpbridge document.pdf --no-raw

# Also save per-page PNG images with bounding box overlays
dmpbridge document.pdf --save-images pdfplumber_images/
```

### Python API

```python
from dmpbridge import process_pdf

# Uses model set in config.py
blocks = process_pdf("document.pdf", output="labeled.json")

# Override model in code
blocks = process_pdf("document.pdf", model="llama3.1:8b", output="labeled.json")

# Also save pdfplumber page images with bounding box overlays
blocks = process_pdf("document.pdf", output="labeled.json", images_dir="pdfplumber")

# Inspect results
from collections import Counter
print(Counter(b["label"] for b in blocks))
# Counter({'content': 130, 'section': 28, 'subsection': 12, 'document_title': 1})
```

### Raw pdfplumber extraction JSON

Every run automatically saves the raw pdfplumber extraction to `data/pdfplumber/<name>.json` **before** LLM labeling. This file contains all blocks with `label: null` — the exact input the LLM receives. Use it to inspect what pdfplumber detected independently of the labeling step.

---

## Part 2 — Evaluation (extraction quality metrics)

The `dmpbridge.evaluation` module compares extracted text against manually curated reference files using nine metrics.

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Word Capture | Share of reference words successfully captured |
| ROUGE-L | Sequence similarity via Longest Common Subsequence |
| Word Precision | Proportion of extracted words that are correct (cleanliness) |
| Word Recall | Proportion of reference words recovered (completeness) |
| Word F1 | Balanced precision / recall score |
| Extracted Word Count | Total words in extraction (useful for spotting over-extraction) |
| Reference Word Count | Baseline denominator for all completeness calculations |
| Missing Word Count | Words in reference absent from extraction |
| Extra Word Count | Words in extraction not in reference (noise) |

### Results on 10 DMP samples

| Sample | Word Capture | ROUGE-L | Precision | Recall | F1 |
|--------|-------------|---------|-----------|--------|----|
| sample1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sample2 | 0.999 | 0.999 | 1.000 | 0.999 | 0.999 |
| sample3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sample4 | 0.999 | 0.999 | 0.998 | 0.999 | 0.999 |
| sample5 | 0.999 | 0.999 | 0.998 | 0.999 | 0.999 |
| sample6 | 0.997 | 0.997 | 0.993 | 0.997 | 0.995 |
| sample7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sample8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sample9 | 0.999 | 0.999 | 0.998 | 0.999 | 0.999 |
| sample10 | 0.993 | 0.993 | 0.997 | 0.993 | 0.995 |

### Python API

```python
from dmpbridge.evaluation.pdfplumber_text_evaluator import evaluate_pdfplumber_text

result = evaluate_pdfplumber_text(
    extracted_txt_path="data/pdfplumber_extracted_text/sample1.txt",
    reference_txt_path="data/reference_text/sample1_reference.txt",
    clean_text=True,   # lowercases and strips punctuation before comparison
)
print(result)
# {
#   'sample_id': 'sample1', 'word_capture': 1.0, 'rouge_l': 1.0,
#   'word_precision': 1.0, 'word_recall': 1.0, 'word_f1': 1.0,
#   'extracted_word_count': 1006, 'reference_word_count': 1006,
#   'missing_word_count': 0, 'extra_word_count': 0
# }
```

---

## Part 3 — Viewer (PDF + JSON side by side)

### Option A — Standalone HTML (no server)

Open `dmpbridge.html` directly in any modern browser. Drag and drop a PDF and JSON file onto the page, or use the Load buttons.

No installation required.

### Option B — FastAPI server

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

uvicorn main:app --reload
```

Open **http://localhost:8000**

Files are uploaded to the server and served back over HTTP.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [01_pdfplumber_batch_test.ipynb](notebooks/01_pdfplumber_batch_test.ipynb) | Run all 10 sample PDFs through pdfplumber, save JSON blocks, plain text, Markdown, and debug CSVs |
| [02_evaluation_pdfplumber_batch_test.ipynb](notebooks/02_evaluation_pdfplumber_batch_test.ipynb) | Evaluate extraction quality for each sample against reference text using all nine metrics |

---

## Workflow end to end

```
1. Run pipeline
   dmpbridge data/pdfsamples/sample1.pdf -v -o data/llmlabeled/sample1_labeled.json
   → data/pdfplumber/sample1.json              (raw pdfplumber extraction)
   → data/llmlabeled/sample1_labeled.json      (LLM-labeled output)

2. Evaluate extraction quality (optional)
   Run notebook 02_evaluation_pdfplumber_batch_test.ipynb
   → compares data/pdfplumber_extracted_text/*.txt against data/reference_text/*_reference.txt

3. Start viewer
   uvicorn main:app --reload
   → http://localhost:8000

4. Load files in viewer
   Load data/pdfsamples/sample1.pdf + data/llmlabeled/sample1_labeled.json
   (or load data/pdfplumber/sample1.json to inspect raw extraction)

5. Inspect & verify
   Click rows ↔ PDF highlights sync automatically
```
