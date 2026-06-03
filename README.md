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
│   ├── reference_pdfs/                      # Original PDF documents
│   │   ├── sample1.pdf
│   │   └── sample10.pdf
│   │
│   ├── reference_text/                      # Reference text for validation
│   │   ├── sample1_reference.txt
│   │   └── sample10_reference.txt
│   │
│   ├── reference_structure_blocks/          # Reference structured blocks for comparison
│   │   ├── sample1_reference.json
│   │   └── sample10_reference.json
│   │
│   ├── pdfplumber_extracted_blocks/         # Structured block extraction (JSON)
│   │   ├── sample1.json
│   │   └── sample10.json
│   │
│   ├── pdfplumber_extracted_blocks_debug/   # Debug output from block extraction
│   │   ├── sample1_debug.json
│   │   └── sample10_debug.json
│   │
│   ├── pdfplumber_extracted_text/           # Raw text extraction
│   │   ├── sample1.txt
│   │   └── sample10.txt
│   │
│   ├── pdfplumber_extracted_markdown/       # Markdown-formatted extraction
│   │   ├── sample1.md
│   │   └── sample10.md
│   │
│   └── llama_structured_blocks/             # LLM-labeled structured data
│       ├── sample1_llama_blocks.json
│       └── sample10_llama_blocks.json
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
│   │   └── llm_narrative_blocks.py          # Narrative block labeling
│   │
│   ├── vision/                              # Vision-based processing (future)
│   │   └── __init__.py
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
│   ├── 03_llama_dmp_narrative_labeling_batch_test.ipynb
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
 
### Running Jupyter Notebooks
 
Start Jupyter and navigate to the `notebooks/` directory:
 
```bash
jupyter notebook
```
 
Then open any of the provided notebooks to explore:
- **01_pdfplumber_batch_test.ipynb** — Batch PDF extraction
- **02_evaluation_pdfplumber_test.ipynb** — Evaluate extraction quality
- **03_llama_dmp_narrative_labeling_batch_test.ipynb** — LLM-based labeling
- **04_evaluation_llama_dmp_narrative_batch_test.ipynb** — Evaluate LLM output
## Configuration
 
Key configuration parameters can be adjusted in `pyproject.toml`:
 
```toml
[tool.dmpbridge]
# Add your configuration settings here
```
 


 