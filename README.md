# DMPBridge

Extract and label the content of Data Management Plan (DMP) PDFs using a local LLM.

---

## Quick start

**Requirements:** Python 3.10+ · [Ollama](https://ollama.com) running locally

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS / Linux

pip install -e .
```

Pull a model:

```bash
ollama pull gemma4:e4b           # ~3 GB  — recommended for most machines
ollama pull llama3.1:8b          # ~5 GB  — good quality, fast
ollama pull llama3.3:70b         # ~42 GB — best accuracy, requires large GPU
```

---

## Run the pipeline

### Option 1 — Notebook (easiest)

Open `notebooks/0-run_pipeline.ipynb`, edit the first config cell to set your PDF and model, then run all cells.

### Option 2 — Python

```python
import dmpbridge

blocks = dmpbridge.process_pdf(
    "path/to/document.pdf",
    model="gemma4:e4b",
    extractor="pdfplumber",
    output="output_labeled.json",
    structured_output="output_structured.json",
)
```

### Option 3 — CLI

```bash
# One PDF
dmpbridge path/to/document.pdf --model gemma4:e4b

# Batch (sample1.pdf … sample10.pdf)
dmpbridge-wholedoc --model gemma4:e4b --extractor pdfplumber
dmpbridge-wholedoc --model llama3.1:8b --start 3 --end 6   # subset
```

Output is written to `data/output/labeled/` by default. Re-runs skip files that already exist.

---

## Output labels

| Label | Meaning |
|---|---|
| `title` | Document title |
| `section.title` | Section heading |
| `section.description` | Instructions written by the funder |
| `question.text` | A question or prompt |
| `answer.text` | The researcher's response |

Each block in the labeled JSON also carries its `page` number and a `confidence` score.

---

## Extractors

| Extractor | Install | Best for |
|---|---|---|
| `pdfplumber` | included | Most PDFs |
| `docling` | `pip install -e ".[docling]"` | Complex layouts |

Pass `extractor="docling"` to `process_pdf()` or `--extractor docling` on the CLI.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```
