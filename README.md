# DMP Bridge
DMP Bridge is an open-source, MIT-licensed, Python-based pipeline that extracts DMP fields and converts them into RDA Common Standard JSON with DMPTool extensions..

## Repository Structure
```text
dmpbridge/
│
├── data/
│   ├── reference_pdfs/
│   │   ├── sample1.pdf
│   │   └── sample6.pdf
│   │
│   ├── pdfplumber_extracted_blocks/
│   │   ├── sample1.json
│   │   └── sample6.json
│   │
│   ├── pdfplumber_extracted_text/
│   │   ├── sample1.txt
│   │   └── sample6.txt
│   │
│   ├── pdfplumber_extracted_markdown/
│   │   ├── sample1.md
│   │   └── sample6.md
│   │
│   └── reference_text/
│       ├── sample1_reference.txt
│       └── sample6_reference.txt
│
├── src/
│   └── dmpbridge/
│       ├── __init__.py
│       │
│       ├── pdf/
│       │   ├── __init__.py
│       │   └── pdfplumber_extractor.py
│       │   
│       │
│       ├── vision/
│       │   ├── __init__.py
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── pdfplumber_text_evaluator.py                   
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── file_io.py
│
├── notebooks/
│   ├── 01_pdfplumber_batch_test.ipynb
│  
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
