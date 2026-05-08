# DMP Bridge
DMP Bridge is an open-source, MIT-licensed, Python-based pipeline that extracts DMP fields and converts them into RDA Common Standard JSON with DMPTool extensions.

## Repository Structure
```text
dmpbridge/
│
├── data/
│   ├── raw_pdfs/                  # input PDFs 
│   │
│   ├── page_images/              # PDF → images (for Qwen2-VL)
│   │   └── {pdf_name}/page_1.png
│   │
│   ├── pdfplumber_blocks/        # raw extracted text + layout
│   │   └── {pdf_name}.json
│   │
│   ├── qwen_outputs/             # detected headers/structure
│   │   └── {pdf_name}.json
│   │
│   ├── markdown/                 # reconstructed Markdown
│   │   └── {pdf_name}.md
│   │
│   └── structure_json/           # FINAL output of phase 1
│       └── {pdf_name}.json
│
├── src/
│   └── dmpbridge/
│       ├── __init__.py
│       │
│       ├── pdf/
│       │   ├── __init__.py
│       │   ├── pdf_type_detector.py                  # optional, keep for later
│       │   ├── docling_extractor.py
│       │   ├── docling_postprocessor.py
│       │   ├── pdfplumber_extractor.py
│       │   └── page_image_converter.py
│       │
│       ├── vision/
│       │   ├── __init__.py
│       │   └── qwen_structure_detector.py
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── block_fusion.py                     # later, for rule + Qwen fusion
│       │   ├── structure_detector.py
│       │   ├── markdown_builder.py                 # optional, later
│       │   └── structure_json_builder.py
│       │   └── text_cleaner.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── extraction_evaluator.py                     # later
│       │   └── header_evaluator.py                         # later
│       │
│       └── utils/
│       │   ├── __init__.py
│           ├── logger.py
│           └── file_io.py
│
├── notebooks/
│   ├── 01_pdfplumber_test.ipynb
│   ├── 02_qwen_test.ipynb
│   ├── 03_fusion_test.ipynb
│   └── 04_structure_output.ipynb
│
├── outputs/
│   ├── debug/
│   ├── logs/
│   └── reports/
│
├── schemas/
│   └── rda_dmp_dmptool_extension_skeleton.json     # your intermediate JSON schema
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
└── README.md


```
## Setup (Local Development)

### Step 1 — Clone the repository
```bash
git clone https://github.com/fairdataihub/dmpbridge.git
cd dmpbridge
code .
```

### Step 2 — Create and activate a virtual environment

**Windows (cmd):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
# or (recommended for local dev)
pip install -e .
```
