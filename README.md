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

Open `notebooks/01-run_pipeline.ipynb`, edit the first config cell to set your PDF and model, then run all cells.

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

Re-runs skip samples that already have output.

---

## Output layout

The pipeline runs in four stages, each writing its own folder so any stage can be
inspected or diffed without re-running the others:

```
data/output/
├── 1_extracted/<extractor>/sampleN.json     text blocks, no labels
├── 2_labeled/<tag>/sampleN.json             the same blocks, labeled by the LLM
├── 3_structured/<tag>/sampleN.json          nested into the DMP Tool schema
└── 4_final/<tag>/sampleN.json               annotation rules applied
```

`<tag>` is `<model>_<extractor>_whole_doc`. Filenames are identical at every stage,
so you can open `sampleN.json` in each folder and follow one document through.

**Stage 1 is keyed by extractor, not by model**, and is cached. Extraction doesn't
depend on which LLM labels the blocks, so labeling a corpus with three models costs
one extraction rather than three — worth several minutes per sweep with `lighton`,
which runs a vision model over every page.

```bash
dmpbridge-wholedoc --model llama3.1:8b --extractor lighton   # extracts, caches
dmpbridge-wholedoc --model gemma4:e4b  --extractor lighton   # reuses the cache

dmpbridge-wholedoc --model gemma4:e4b --extractor lighton --no-cache   # force re-extract
dmpbridge-wholedoc --model gemma4:e4b --extractor lighton --no-rules   # skip stage 4
```

To extract only, with no LLM involved:

```bash
python scripts/extract_pdfplumber.py            # merged (default)
python scripts/extract_pdfplumber.py --raw      # line-level, no merging
python scripts/extract_pdfplumber.py --both     # both, with a comparison
```

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
| `lighton` | `pip install -e ".[lighton]"` | Scanned / image-based PDFs (OCR) |

`pdfplumber` reads a PDF line by line, so a wrapped paragraph arrives as several
blocks. Those are merged back into paragraphs before labeling, bringing it to the
same granularity as the other two (roughly 25 blocks per document instead of 74).
Pass `merge_lines=False` to get the raw line-level blocks:

```python
from dmpbridge.extractors import get_extractor
get_extractor("pdfplumber", merge_lines=False)
```

Pass the extractor name to `process_pdf()` or the CLI:

```python
dmpbridge.process_pdf("document.pdf", model="gemma4:e4b", extractor="lighton")
```

```bash
dmpbridge-wholedoc --model gemma4:e4b --extractor lighton
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```
