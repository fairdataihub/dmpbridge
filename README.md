# dmpbridge

A two-part tool for analyzing PDF structure:

1. **Pipeline** — extract text from a PDF with pdfplumber, classify each block as `document_title`, `section`, `subsection`, or `content` using a local LLM (via Ollama), and output a labeled JSON file.
2. **Viewer** — a side-by-side PDF + JSON browser UI that overlays bounding boxes on the PDF, synchronized with the JSON table.

---

## Project structure

```
dmpbridge/
├── __init__.py        # exports process_pdf
├── extractor.py       # pdfplumber text extraction + page image export
├── classifier.py      # Ollama LLM classifier
├── pipeline.py        # combines extraction + classification
├── cli.py             # dmpbridge command-line tool
└── config.py          # ← edit here to change model / host / batch size

data/
├── pdfsamples/        # sample PDFs for testing
├── llmlabeled/        # LLM-generated labeled JSON output
├── manuallabeled/     # manually corrected labeled JSON
└── pdfplumber/        # (generated) per-page PNG images with block overlays

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
ollama pull deepseek-r1:latest   # reasoning model, thorough but slower
```

---

## Part 1 — Pipeline (PDF → labeled JSON)

### Configure the model

Open **[dmpbridge/config.py](dmpbridge/config.py)** and set your preferred model:

```python
# Change this line to switch models — no other code needs to change
MODEL = "llama3.2:latest"

HOST       = "http://localhost:11434"   # Ollama server URL
BATCH_SIZE = 10                         # blocks per LLM request
```

### CLI usage

```powershell
# Basic — output saved as <name>_labeled.json next to the PDF
dmpbridge document.pdf

# Specify output path
dmpbridge document.pdf -o output/labeled.json

# Override model for this run (ignores config.py)
dmpbridge document.pdf --model llama3.1:8b

# Show detailed progress per batch
dmpbridge document.pdf -v

# Save per-page PNG images to the default "pdfplumber/" folder
dmpbridge document.pdf --save-images

# Save images to a custom folder
dmpbridge document.pdf --save-images data/pdfplumber
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

### Image export (pdfplumber page images)

Passing `--save-images` renders every page as a PNG with color-coded bounding boxes saved to the `pdfplumber/` folder (one file per page: `page_001.png`, `page_002.png`, …).

**Color coding matches the viewer:**

| Label | Box color |
|---|---|
| `document_title` | Purple |
| `section` | Gold |
| `subsection` | Teal |
| `content` | Blue |

**Dependencies** — image export requires only `Pillow`, which is already pulled in by `pdfplumber>=0.10` (via `pypdfium2`). No extra system installs needed.

---

### Testing different models

```powershell
# Test with llama3.2 (default, fast)
dmpbridge data/pdfsamples/sample2.pdf -o data/llmlabeled/sample2_llama32.json

# Test with llama3.1:8b (more accurate)
dmpbridge data/pdfsamples/sample2.pdf --model llama3.1:8b -o data/llmlabeled/sample2_llama31.json

```

### Output JSON format

Each block in the output has:

| Field | Type | Description |
|---|---|---|
| `page` | int | Page number (1-based) |
| `line_order` | int | Line index on the page |
| `text` | string | Extracted text content |
| `x0` | float | Left edge in points |
| `top` | float | Top edge in points (from top of page) |
| `x1` | float | Right edge in points |
| `bottom` | float | Bottom edge in points |
| `avg_font_size` | float | Average font size |
| `font_names` | list[str] | Font names used in the line |
| `is_bold` | bool | Whether text is bold |
| `label` | string | `document_title` · `section` · `subsection` · `content` |

Example:

```json
[
  {
    "page": 1, "line_order": 1,
    "text": "Annual Report 2024",
    "x0": 72.0, "top": 80.0, "x1": 400.0, "bottom": 102.0,
    "avg_font_size": 24.0,
    "font_names": ["ABCDEF+TimesNewRoman,Bold"],
    "is_bold": true,
    "label": "document_title"
  },
  {
    "page": 1, "line_order": 3,
    "text": "1. Introduction",
    "x0": 72.0, "top": 130.0, "x1": 200.0, "bottom": 148.0,
    "avg_font_size": 14.0,
    "font_names": ["ABCDEF+TimesNewRoman,Bold"],
    "is_bold": true,
    "label": "section"
  }
]
```

---

## Part 2 — Viewer (PDF + JSON side by side)

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

### Viewer controls

| Control | Action |
|---|---|
| **Load PDF / Load JSON** | File picker or drag & drop |
| **‹ / ›** | Previous / next PDF page |
| **Zoom** | 75 % – 250 % |
| **Selected / All on page / None** | Bounding box overlay mode |
| **Search box** | Filter by text content |
| **Page / Style / Font / Label dropdowns** | Filter the table |
| Click table header | Sort by that column |
| Click a table row | Jump to that location in the PDF |
| Click on the PDF | Select the nearest JSON entry |
| **↑ / ↓ arrow keys** | Move selection up / down |
| **← / → arrow keys** | Navigate PDF pages |

### Label colors

| Label | Color |
|---|---|
| `document_title` | Purple |
| `section` | Gold |
| `subsection` | Teal |
| `content` | Dim gray |

---

## Workflow end to end

```
1. Run pipeline
   dmpbridge data/pdfsamples/sample1.pdf -v -o data/llmlabeled/sample1_labeled.json

2. Start viewer
   uvicorn main:app --reload
   → http://localhost:8000

3. Load files in viewer
   Load data/pdfsamples/sample1.pdf + data/llmlabeled/sample1_labeled.json

4. Inspect & verify
   Click rows ↔ PDF highlights sync automatically
```
