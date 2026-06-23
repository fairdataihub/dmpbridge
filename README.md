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
├── pipeline.py     # combines extraction + classification + smoothing
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
├── 01_pdfplumber_batch_test.ipynb          # batch extraction across all sample PDFs
├── 02_evaluation_pdfplumber_batch_test.ipynb
└── 03_label_evaluation.ipynb               # visual evaluation: confusion matrix + F1 charts + model comparison

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

### How it works

```
PDF
 ↓ pdfplumber
Line-level text blocks  (page, coordinates, font size, bold, italic)
   · bold/italic determined from the first non-whitespace character's font
   · duplicate characters (layered bold shadow) collapsed by deduplication
 ↓ saved to data/pdfplumber/<name>.json  (raw, before labeling)
 ↓ Ollama LLM — batched in groups of 10
   · Few-shot examples from manually labeled DMPs guide each label
   · Last 3 labeled blocks are sent as context for each new batch
Labeled blocks: title | section.title | section.description | question.text | answer.text
 ↓ post-processing smoothing rules (pipeline.py)
   · Rule 1: bold+italic unnumbered block mislabeled section.title/section.description → question.text
   · Rule 2: italic-only block following question.text mislabeled section.description → question.text
 ↓ saved to data/llmlabeled/<name>_<model>.json
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

# Also produce hierarchical structured JSON (DMP Tool narrative schema)
dmpbridge document.pdf -o data/llmlabeled/sample1_llama3.3-70b.json --structured

# Override model for this run (output is named after the model automatically)
dmpbridge document.pdf --model llama3.1:8b -o data/llmlabeled/sample1_llama3.1-8b.json

# Show detailed progress
dmpbridge document.pdf -v

# Skip saving the raw pdfplumber JSON
dmpbridge document.pdf --no-raw
```

### Python API

```python
from dmpbridge import process_pdf

# Flat output only (default)
blocks = process_pdf("document.pdf", output="labeled.json")

# Both flat + structured in one call
blocks = process_pdf(
    "document.pdf",
    output="labeled.json",
    structured_output="labeled_structured.json",
)

# Override model
blocks = process_pdf("document.pdf", model="llama3.1:8b", output="labeled.json")

# Inspect label counts
from collections import Counter
print(Counter(b["label"] for b in blocks))
# Counter({'answer.text': 52, 'question.text': 18, 'section.description': 8,
#          'section.title': 4, 'title': 1})
```

### Convert an existing flat file to structured

```python
from dmpbridge import convert_file

# Writes sample1_llama3.3-70b_structured.json next to the input
convert_file("data/llmlabeled/sample1_llama3.3-70b.json")

# With a URL pointing to the source PDF (stored in download_url)
convert_file(
    "data/llmlabeled/sample1_llama3.3-70b.json",
    pdf_url="https://example.com/dmps/123/narrative",
)

# Explicit output path
convert_file(
    "data/llmlabeled/sample1_llama3.3-70b.json",
    "data/llmlabeled/sample1_structured.json",
)
```

The structured JSON follows the DMP Tool narrative schema. `id` fields are omitted throughout because they cannot be determined from a PDF:

```json
{
  "narrative": {
    "download_url": "",
    "template": {
      "title": "DATA MANAGEMENT AND SHARING PLAN",
      "description": "",
      "version": "v1",
      "section": [
        {
          "title": "Element 1: Data Type:",
          "description": "",
          "order": 1,
          "question": [
            {
              "text": "A. Types and amount of scientific data...",
              "order": 1,
              "answer": {
                "json": {
                  "type": "textArea",
                  "answer": "This secondary data analysis project...",
                  "meta": {
                    "schemaVersion": "1.0"
                  }
                }
              }
            }
          ]
        }
      ]
    }
  }
}
```

> `id` fields (`template.id`, `section.id`, `question.id`, `answer.id`) are omitted because they
> cannot be determined by reading a PDF. They can be added downstream once the record is stored
> in the DMP Tool database.

---

## Evaluation

Compare LLM output against manually labeled ground truth in `data/manuallabeled/`.

### CLI script

```powershell
# Evaluate all samples — prints per-sample accuracy + confusion matrix + F1
python evaluate.py

# Evaluate a single file
python evaluate.py data/llmlabeled/sample1_llama3.3-70b.json
```

Output files are automatically matched by model name — changing `MODEL` in `dmpbridge/config.py` updates both the pipeline output path and the evaluation target with no other changes needed.

Example output (llama3.3:70b):

```
sample1       total= 78  unmatched= 0  accuracy= 97.4%
sample2       total=171  unmatched= 0  accuracy= 96.5%
sample3       total= 69  unmatched= 0  accuracy= 98.6%
sample4       total= 78  unmatched= 0  accuracy=100.0%
sample5       total= 80  unmatched= 0  accuracy= 98.8%
sample6       total= 24  unmatched= 3  accuracy= 75.0%
sample7       total= 17  unmatched= 0  accuracy= 94.1%
sample8       total= 59  unmatched= 0  accuracy= 96.6%
sample9       total= 85  unmatched= 0  accuracy= 98.8%
sample10      total= 68  unmatched= 0  accuracy= 95.6%

Label                    Precision    Recall        F1   Support
--------------------------------------------------------------
title                       100.0%     81.8%     90.0%        11
section.title                81.8%     90.0%     85.7%        40
section.description          89.1%    100.0%     94.2%        57
question.text                88.4%     92.7%     90.5%        41
answer.text                 100.0%     98.1%     99.0%       577
--------------------------------------------------------------
Overall accuracy                                 97.2%       726
```

`unmatched` counts blocks where no gold entry reaches ≥ 75% token containment — these are counted as errors. The confusion matrix and F1 scores cover only the 726 matched blocks.

### Model comparison

Both models have been evaluated on all 10 samples:

| Sample | llama3.3:70b | llama3.1:8b | Diff |
|--------|-------------|------------|------|
| sample1 | 97.4% | 82.1% | -15.4% |
| sample2 | 96.5% | 62.6% | -33.9% |
| sample3 | 98.6% | 81.2% | -17.4% |
| sample4 | 100.0% | 65.4% | -34.6% |
| sample5 | 98.8% | 67.5% | -31.2% |
| sample6 | 75.0% | 66.7% | -8.3% |
| sample7 | 94.1% | 82.4% | -11.8% |
| sample8 | 96.6% | 69.5% | -27.1% |
| sample9 | 98.8% | 61.2% | -37.6% |
| sample10 | 95.6% | 70.6% | -25.0% |
| **OVERALL** | **96.8%** | **69.0%** | **-27.8%** |

Per-label F1 comparison:

| Label | llama3.3:70b | llama3.1:8b | Diff |
|-------|-------------|------------|------|
| title | 90.0% | 66.7% | -23.3% |
| section.title | 85.7% | 63.9% | -21.8% |
| section.description | 94.2% | 40.0% | -54.2% |
| question.text | 90.5% | 33.8% | -56.7% |
| answer.text | 99.0% | 80.2% | -18.8% |

`llama3.1:8b` collapses on `question.text` (F1 33.8%) and `section.description` (F1 40.0%) — the labels requiring understanding of funder-written instructions vs researcher-written sub-questions. `llama3.3:70b` is required for reliable results.

### Notebook (visual)

Open `notebooks/03_label_evaluation.ipynb` for interactive charts:

```powershell
jupyter lab notebooks/03_label_evaluation.ipynb
```

The notebook shows:
- Per-sample accuracy bar chart
- Confusion matrix heatmaps (raw counts + row-normalised recall)
- Precision / Recall / F1 grouped bar chart per label
- Full table of mislabeled blocks with their text
- Drill-down cell to inspect any specific confusion pair
- **Section 7 — Model comparison:** side-by-side grouped bar charts and 2×2 confusion matrix grid comparing llama3.3:70b vs llama3.1:8b across all samples and all labels

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
1. Run the pipeline  (output named after the model automatically)
   dmpbridge data/pdfsamples/sample1.pdf -o data/llmlabeled/sample1_llama3.3-70b.json --structured

2. Evaluate against ground truth
   python evaluate.py data/llmlabeled/sample1_llama3.3-70b.json

3. Compare models (run with a second model — results saved separately)
   dmpbridge data/pdfsamples/sample1.pdf --model llama3.1:8b \
             -o data/llmlabeled/sample1_llama3.1-8b.json --structured

4. Open notebook for visual comparison
   jupyter lab notebooks/03_label_evaluation.ipynb
   → Sections 1–6: single-model analysis (whichever model is in config.py)
   → Section 7:    side-by-side model comparison charts

5. Open the viewer
   Open dmpbridge.html in a browser  (or run: uvicorn main:app --reload)

6. Load files
   Load data/pdfsamples/sample1.pdf + data/llmlabeled/sample1_llama3.3-70b.json

7. Inspect
   Click any row in the table → the corresponding block highlights in the PDF
   Use the Label filter to show only sections, questions, or answers
```
