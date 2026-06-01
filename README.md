# DMP Bridge
DMP Bridge is an open-source, MIT-licensed, Python-based pipeline that extracts DMP fields and converts them into RDA Common Standard JSON with DMPTool extensions..

## Repository Structure
```text
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
│   ├── llama_structured_blocks/
│   │   ├── sample1_llama_blocks.json
│   │   └── sample6_llama_blocks.json
│   │
│   ├── llama_narrative_json/
│   │   ├── sample1_llama_narrative.json
│   │   └── sample6_llama_narrative.json
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
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── llama_client.py
│       │   ├── llm_narrative_blocks.py
│       │   ├── dmp_prompt_builder.py
│       │   └── generate_json.py
│       │
│       ├── vision/
│       │   ├── __init__.py
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── text_cleaner.py
│       │   └── structure_json_builder.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── pdfplumber_text_evaluator.py
│       │   └── narrative_json_evaluator.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── file_io.py
│
├── notebooks/
│   ├── 01_pdfplumber_batch_test.ipynb
│   ├── 02_evaluation_pdfplumber_test.ipynb
│   ├── 03_llama_narrative_structure_test.ipynb
│   └── 04_narrative_json_evaluation.ipynb
│
├── outputs/
│   ├── debug/
│   ├── logs/
│   └── reports/
│
├── schemas/
│   └── rda_dmp_dmptool_extension_skeleton.json
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Current Pipeline

```text
PDF
↓
PDFPlumber Extraction
↓
Text Cleaning
↓
Markdown
↓
Llama 3.1 8B (Ollama)
↓
Structured Blocks
↓
Narrative JSON Builder
↓
DMPTool-Compatible Narrative JSON
```



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
