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
└── 03_label_evaluation.ipynb               # visual evaluation: confusion matrix + F1 charts

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

Example output:

```
sample1       total= 78  correct=76    unmatched= 0  accuracy= 97.4%
sample2       total=171  correct=165   unmatched= 0  accuracy= 96.5%
sample3       total= 69  correct=68    unmatched= 0  accuracy= 98.6%
sample4       total= 78  correct=78    unmatched= 0  accuracy=100.0%
sample5       total= 80  correct=79    unmatched= 0  accuracy= 98.8%
sample6       total= 24  correct=18    unmatched= 3  accuracy= 75.0%
sample7       total= 17  correct=16    unmatched= 0  accuracy= 94.1%
sample8       total= 59  correct=57    unmatched= 0  accuracy= 96.6%
sample9       total= 85  correct=84    unmatched= 0  accuracy= 98.8%
sample10      total= 68  correct=65    unmatched= 0  accuracy= 95.6%

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

`unmatched` counts blocks where no gold entry reaches ≥ 75% token containment — these are counted as errors in the per-sample accuracy. The confusion matrix and F1 scores cover only the 726 blocks that were successfully matched to a gold label.

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
   dmpbridge data/pdfsamples/sample1.pdf -o data/llmlabeled/sample1_llama3.3-70b.json

2. Convert to hierarchical structured JSON (same schema as manual annotations)
   python -c "from dmpbridge import convert_file; convert_file('data/llmlabeled/sample1_llama3.3-70b.json')"
   → writes data/llmlabeled/sample1_llama3.3-70b_structured.json

3. Evaluate against ground truth
   python evaluate.py data/llmlabeled/sample1_llama3.3-70b.json

4. Open the viewer
   Open dmpbridge.html in a browser  (or run: uvicorn main:app --reload)

5. Load files
   Load data/pdfsamples/sample1.pdf + data/llmlabeled/sample1_llama3.3-70b.json

6. Inspect
   Click any row in the table → the corresponding block highlights in the PDF
   Use the Label filter to show only sections, questions, or answers
```
