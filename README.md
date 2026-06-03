# DMP Bridge
 
An open-source Python pipeline for extracting Data Management Plan (DMP) fields from PDF documents and converting them into **RDA Common Standard JSON** with DMPTool extensions.
 
## Features
 
- **PDF Extraction**: Extract structured content from DMP PDFs using pdfplumber
- **LLM-Powered Processing**: Leverage Llama models for intelligent narrative block labeling
- **Text Cleaning**: Automated text normalization and preprocessing
- **RDA Compliance**: Convert extracted data to RDA Common Standard JSON format
- **DMPTool Extensions**: Support for DMPTool-specific extensions and custom fields
- **Evaluation Framework**: Built-in tools for validating extraction accuracy
- **Modular Architecture**: Clean separation of concerns with dedicated modules for each processing stage
## Repository Structure
 
```
dmpbridge/
├── data/                                    # Sample data and extraction outputs
│
├── src/dmpbridge/                           # Main package source code
│   ├── __init__.py
│   │
│   ├── pdf/                                 # PDF extraction module
│   │   ├── __init__.py
│   │   └── pdfplumber_extractor.py          # pdfplumber-based PDF parser
│   │
│   ├── llm/                                 # LLM integration module
│   │   ├── __init__.py
│   │   ├── llama_client.py                  # Llama model client
│   │   └── llm_narrative_blocks_plumberjson.py          # Narrative block labeling
│   │
│   │
│   ├── processing/                          # Data processing module
│   │   ├── __init__.py
│   │   ├── text_cleaner.py                  # Text normalization and cleanup
│   │   └── structure_json_builder.py        # JSON structure conversion
│   │
│   ├── evaluation/                          # Evaluation framework
│   │   ├── __init__.py
│   │   ├── pdfplumber_text_evaluator.py     # Text extraction validation
│   │   └── narrative_json_evaluator.py      # LLM output validation
│   │
│   └── utils/                               # Utility functions
│       ├── __init__.py
│       ├── logger.py                        # Logging configuration
│       └── file_io.py                       # File I/O operations
│
├── notebooks/                               # Jupyter notebooks for testing
│   ├── 01_pdfplumber_batch_test.ipynb       # PDF extraction batch processing
│   ├── 02_evaluation_pdfplumber_test.ipynb  # Text extraction evaluation
│   ├── 03_llama_narrative_labeling_plumberjson_batch_test.ipynb
│   └── 04_evaluation_llama_dmp_narrative_batch_test.ipynb
│
├── outputs/                                 # Generated outputs
│   ├── debug/                               # Debug information
│   ├── logs/                                # Application logs
│   └── reports/                             # Evaluation reports
│
├── schemas/                                 # JSON schemas
│   └── rda_dmp_dmptool_extension_skeleton.json
│
├── tests/                                   # Unit and integration tests
│
├── requirements.txt                         # Python dependencies
├── pyproject.toml                           # Package configuration
└── README.md
```
 
## Quick Start
 
### Prerequisites
 
- Python 3.8 or higher
- pip package manager
- Git
### Setup (Local Development)
 
#### Step 1: Clone the Repository
 
```bash
git clone https://github.com/fairdataihub/dmpbridge.git
cd dmpbridge
```
 
#### Step 2: Create and Activate Virtual Environment
 
**Windows (cmd):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```
 
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
 
**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```
 
#### Step 3: Install Dependencies
 
```bash
# Standard installation
pip install -r requirements.txt
 
# Recommended for local development (editable mode)
pip install -e .
```
 
## Usage
 
### Basic PDF Extraction
 
```python
from dmpbridge.pdf import pdfplumber_extractor
 
# Extract text from a PDF
extractor = pdfplumber_extractor.PDFExtractor()
text = extractor.extract_text("path/to/dmp.pdf")
```


 