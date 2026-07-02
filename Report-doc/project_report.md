# DMPBridge — Project Status Report

**Date:** 2026-07-02  
**Version:** 0.1.0  
**Author:** Nahid Zeinali

---

## 1. Project Overview

DMPBridge is a Python pipeline that reads Data Management Plan (DMP) PDF documents and
classifies every text block into one of five semantic labels. The output is a flat labeled
JSON file and an optional hierarchical "structured" JSON following the DMP Tool narrative
schema.

**Core problem:** DMP PDFs mix funder-written template text with researcher-written
responses. Automated systems that process DMPs need to distinguish these reliably.
DMPBridge treats this as a block-level multi-class classification task, leveraging LLMs
as the classifier.

---

## 2. Label Schema

Every text block extracted from a PDF is assigned exactly one of five labels:

| Label | Meaning |
|---|---|
| `title` | Single main document title; appears once per document |
| `section.title` | Numbered or named section heading (e.g. "1. Data sharing") |
| `section.description` | Funder template text — instructions to the author; uses "should", "must" |
| `question.text` | Sub-question prompt inside a section; asks the author to address a specific topic |
| `answer.text` | Researcher's actual written response — narrative paragraphs describing plans |

The key distinction: `section.description` is written by the funder; `answer.text` is
written by the researcher. `question.text` is a specific sub-prompt, not a section header.

---

## 3. Pipeline Architecture

```
PDF file
   │
   ▼
[Extraction]  dmpbridge.preprocess.extract_blocks()
   pdfplumber reads the PDF page by page, line by line.
   Each line becomes one block dict (12 fields):
     page, line_order, text, x0, top, x1, bottom,
     avg_font_size, font_names, is_bold, is_italic, label
   text_cleaner removes duplicate characters (layered PDF rendering artefact).
   ~60–90 blocks per DMP document.
   │
   ▼
[Classification]  LLM strategy (batch / whole_doc / pdf_direct)
   Blocks are sent to an LLM with a structured system prompt + few-shot examples.
   The LLM returns a JSON array mapping block IDs → label strings.
   │
   ▼
[Output]  Flat JSON + Structured JSON
   data/output/labeled/{tag}/sampleN.json        — flat list of labeled blocks
   data/output/labeled/{tag}/sampleN_structured.json — hierarchical DMP Tool schema
```

### 3.1 Subpackage Map

```
dmpbridge/
├── core/
│   ├── config.py          — PROVIDER, MODEL, HOST, BATCH_SIZE, API keys
│   ├── pipeline.py        — process_pdf() orchestration
│   └── converter.py       — to_structured() → DMP Tool JSON schema
├── preprocess/
│   ├── pdfplumber_reader.py  — extract_blocks(); pdfplumber line extraction
│   ├── text_cleaner.py       — clean_blocks(), deduplicate doubled chars
│   └── page_images.py        — optional per-page PNG export with bounding boxes
├── strategies/
│   ├── batch.py           — BatchStrategy: sliding-window batch calls
│   ├── wholedoc.py        — WholeDocStrategy: single call with all blocks
│   └── pdf_direct.py      — PdfDirectStrategy: raw PDF bytes → Claude API
├── models/
│   ├── ollama.py          — OllamaModel (local server, structured output)
│   ├── anthropic.py       — AnthropicModel (Anthropic messages API)
│   ├── openai.py          — OpenAIModel (OpenAI chat completions)
│   └── gemini.py          — GeminiModel (Google Gemini API)
├── prompts/
│   ├── system.py          — shared SYSTEM_PROMPT with few-shot examples
│   └── labels.py          — LABELS tuple, OUTPUT_SCHEMA for structured output
├── parsers/
│   └── json_parser.py     — parse_llm_json(): strips markdown fences, repairs JSON
├── evaluation/
│   ├── evaluate.py        — extract_gold, evaluate_sample, gold_metrics, load_method
│   └── experiment.py      — ExperimentConfig, Experiment, YAML-driven runner
├── cli/
│   ├── main.py            — dmpbridge (single PDF, batch strategy)
│   └── wholedoc_cmd.py    — dmpbridge-wholedoc (single PDF, wholedoc strategy)
└── utils/
    ├── logger.py          — get_logger, setup_logging
    └── exceptions.py      — DmpBridgeError, ConfigurationError
```

---

## 4. Labeling Strategies

Three strategies share the same label schema and evaluation framework but differ in
how they send content to the model.

### 4.1 Batch Strategy (`batch`)

**How it works:**
1. Extract ~70 line-level blocks from the PDF via pdfplumber.
2. Divide blocks into windows of `batch_size` (default: 10) with a sliding context
   of 3 already-labeled blocks prepended to each window.
3. Make `ceil(N / batch_size)` LLM calls; merge labels back by block index.

**Prompt payload per call:**
```json
[{"id": 5, "text": "...", "bold": true, "italic": false, "page": 1}, ...]
```
The preceding labeled blocks are included as context so the model can track
structural continuity across window boundaries.

**Advantages:** Fits in any model's context window; supports all four providers.  
**Disadvantages:** Multiple API calls per document; boundary effects possible.

**Config example:**
```yaml
strategy: batch
batch_size: 10
context_size: 3
```

**CLI:** `dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml`

---

### 4.2 Whole-Document Strategy (`wholedoc`)

**How it works:**
1. Extract all blocks from the PDF via pdfplumber (same as batch).
2. Send the entire block list to the model in a single API call.
3. Parse the response JSON array (all N labels at once).

**Advantages:** The model sees the full document structure; no boundary effects.  
**Disadvantages:** High token usage; requires large context window (max_tokens=16384);
only Anthropic and Ollama providers supported.

**CLI:** `dmpbridge-experiment experiments/claude-opus-4-8-wholedoc.yaml`  
Also available as `dmpbridge-wholedoc document.pdf`

---

### 4.3 PDF-Direct Strategy (`pdf_direct`)

**How it works:**
1. Read the raw PDF bytes and base64-encode them.
2. Send to Claude's document API in a single message with the classification prompt.
3. Claude simultaneously extracts and classifies the content — no pdfplumber at all.

**Key difference from batch/wholedoc:** Produces paragraph-level blocks, not line-level
blocks. Typical block count is ~20 per document vs ~70 for pdfplumber strategies.
Bounding-box coordinates are not available.

**Comparability note:** Because paragraph-level blocks cannot be compared
index-by-index with line-level blocks, cross-strategy evaluation uses *gold-based
accuracy* (`gold_metrics()`) rather than block accuracy. Gold accuracy uses the
741 manually labeled gold items as the denominator, making all strategies comparable.

**Only Anthropic is supported** — this strategy relies on Claude's PDF vision capability.

**CLI:** `dmpbridge-pdf --model claude-opus-4-8`

---

## 5. Supported Providers and Models

| Provider | Key Setting | Models Used in Experiments |
|---|---|---|
| Ollama (local) | `DMPBRIDGE_HOST` (default: `http://localhost:11434`) | `llama3.1:8b`, `llama3.3:70b` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-8` |
| OpenAI | `OPENAI_API_KEY` | (configured, not yet used in experiments) |
| Gemini | `GEMINI_API_KEY` | (configured, not yet used in experiments) |

Configuration is set in `dmpbridge/core/config.py` or via environment variables / `.env`:

```bash
DMPBRIDGE_PROVIDER=anthropic
DMPBRIDGE_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 6. Experiments Registry

Seven experiments have been run across three models and three strategies. Each
experiment is defined by a YAML config in `experiments/` and outputs to
`data/output/labeled/{tag}/`.

| ID | Name | Model | Strategy | Provider | Tag | Output Dir |
|---|---|---|---|---|---|---|
| M01 | Llama 3.1-8B Batch | llama3.1:8b | batch | Ollama | `llama3.1-8b_batch` | `data/output/labeled/llama3.1-8b_batch/` |
| M02 | Llama 3.1-8B Whole-Doc | llama3.1:8b | wholedoc | Ollama | `llama3.1-8b_whole_doc` | `data/output/labeled/llama3.1-8b_whole_doc/` |
| M03 | Llama 3.3-70B Batch | llama3.3:70b | batch | Ollama | `llama3.3-70b_batch` | `data/output/labeled/llama3.3-70b_batch/` |
| M04 | Llama 3.3-70B Whole-Doc | llama3.3:70b | wholedoc | Ollama | `llama3.3-70b_whole_doc` | `data/output/labeled/llama3.3-70b_whole_doc/` |
| M05 | Opus 4.8 Batch | claude-opus-4-8 | batch | Anthropic | `claude-opus-4-8_batch` | `data/output/labeled/claude-opus-4-8_batch/` |
| M06 | Opus 4.8 Whole-Doc | claude-opus-4-8 | wholedoc | Anthropic | `claude-opus-4-8_whole_doc` | `data/output/labeled/claude-opus-4-8_whole_doc/` |
| M07 | Opus 4.8 PDF-Direct | claude-opus-4-8 | pdf_direct | Anthropic | `claude-opus-4-8_pdf` | `data/output/labeled/claude-opus-4-8_pdf/` |

### 6.1 Experiment YAML Files

```
experiments/
├── llama3.1-8b-batch.yaml
├── llama3.1-8b-wholedoc.yaml
├── llama3.3-70b-batch.yaml
├── llama3.3-70b-wholedoc.yaml
├── claude-opus-4-8-batch.yaml
├── claude-opus-4-8-wholedoc.yaml
└── claude-opus-4-8-pdf-direct.yaml
```

**Replay any experiment:**
```bash
dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml
dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml --evaluate
dmpbridge-experiment --list
```

---

## 7. Evaluation Framework

### 7.1 Gold Standard

10 sample PDFs, each with a manually annotated JSON file:
- Annotation files: `data/input/ground_truth/sampleN_dmp.json`
- Reference text files: `data/input/ground_truth/reference_text/sampleN_reference.txt`
- Total gold items across all 10 samples: **741 labeled spans**

### 7.2 Matching Algorithm

`evaluate_sample(pred_path, gold_pairs)` uses token-level containment:

1. **Forward check** — for each predicted block, find the best-matching gold item
   using `containment(block_tokens, gold_tokens) ≥ 0.75` threshold.
2. **Reverse check** — for each gold item, mark it as `__missed__` if no predicted
   block covers it with containment ≥ 0.75.

This handles the mismatch between line-level pdfplumber blocks and paragraph-level
gold items.

### 7.3 Two Accuracy Measures

| Measure | Formula | When to use |
|---|---|---|
| Block accuracy | `correct_blocks / total_predicted_blocks` | Comparing batch vs whole-doc within a model |
| Gold accuracy | `correctly_covered_gold / total_gold_items` | Cross-strategy comparison; used for M07 (pdf-direct) |

`gold_metrics(confusion)` returns `(correct, covered, total_gold)` — makes pdf-direct
comparable to pdfplumber-based strategies despite different block counts.

### 7.4 Known Results

| Experiment | Block Accuracy | Gold Accuracy | Notes |
|---|---|---|---|
| M07 — Opus 4.8 PDF-Direct | ~95.1% | ~99.5% | Paragraph-level; no pdfplumber |
| Other experiments | See eval notebooks | — | Loaded from `data/output/labeled/` |

Full per-label precision/recall/F1 and confusion matrices are in the evaluation notebooks.

---

## 8. Data Folder Structure

```
data/
├── input/
│   ├── pdfs/
│   │   └── sample1.pdf … sample10.pdf      (10 DMP documents)
│   └── ground_truth/
│       ├── sample1_dmp.json … sample10_dmp.json   (manual annotations)
│       └── reference_text/
│           └── sample1_reference.txt … sample10_reference.txt
└── output/
    ├── extracted/
    │   └── sample1.json … sample10.json    (pdfplumber blocks, no labels)
    └── labeled/
        ├── llama3.1-8b_batch/
        ├── llama3.1-8b_whole_doc/
        ├── llama3.3-70b_batch/
        ├── llama3.3-70b_whole_doc/
        ├── claude-opus-4-8_batch/
        ├── claude-opus-4-8_whole_doc/
        └── claude-opus-4-8_pdf/
            └── sample1.json … sample10.json
            └── sample1_structured.json … sample10_structured.json
```

---

## 9. Notebooks

Seven numbered notebooks in `notebooks/` cover the full pipeline from raw PDF to
cross-model comparison.

| Notebook | Purpose |
|---|---|
| [000_experiment_log.ipynb](../notebooks/000_experiment_log.ipynb) | Experiment registry, strategy descriptions, cross-model summary table |
| [001_pdfplumber_extraction.ipynb](../notebooks/001_pdfplumber_extraction.ipynb) | Run pdfplumber on all 10 PDFs; inspect block counts, font sizes, bold/italic patterns |
| [002_pdfplumber_extraction_eval.ipynb](../notebooks/002_pdfplumber_extraction_eval.ipynb) | Compare extracted text against reference text; word capture, ROUGE-L, F1, precision, recall |
| [003_strategy_comparison.ipynb](../notebooks/003_strategy_comparison.ipynb) | Side-by-side accuracy across all 7 experiments; confusion matrices; cross-model agreement (Section 7) |
| [004_llama3.1-8b.ipynb](../notebooks/004_llama3.1-8b.ipynb) | Deep-dive eval for llama3.1:8b (batch + whole-doc) |
| [005_llama3.3-70b.ipynb](../notebooks/005_llama3.3-70b.ipynb) | Deep-dive eval for llama3.3:70b (batch + whole-doc) |
| [006_claude-opus-4-8.ipynb](../notebooks/006_claude-opus-4-8.ipynb) | Deep-dive eval for claude-opus-4-8 (batch + whole-doc + pdf-direct) |

**Notes:**
- Notebook `003` loads all 7 variants; pdf-direct is excluded from Section 7
  (cross-model block alignment) because paragraph-level blocks cannot be aligned
  index-by-index with line-level blocks.
- All notebooks were verified to execute cleanly with `jupyter nbconvert --execute`.

---

## 10. CLI Reference

| Command | Strategy | Description |
|---|---|---|
| `dmpbridge document.pdf` | batch | Single PDF, all providers |
| `dmpbridge-wholedoc document.pdf` | wholedoc | Single PDF, Anthropic or Ollama |
| `dmpbridge-pdf --model claude-opus-4-8` | pdf_direct | All PDFs in `data/input/pdfs/`, Anthropic only |
| `dmpbridge-experiment experiments/X.yaml` | any | Run full experiment from YAML config |
| `dmpbridge-experiment experiments/X.yaml --evaluate` | any | Run + print accuracy summary |
| `dmpbridge-experiment --list` | — | List all available experiment configs |
| `dmpbridge-evaluate` | — | Evaluate all samples against the default tag |
| `dmpbridge-evaluate data/output/labeled/tag/sample1.json` | — | Evaluate one file |

---

## 11. System Prompt Design

All strategies share the same system prompt (`dmpbridge/prompts/system.py`). It uses:

1. **Clear label definitions** — each label has a one-line description with key signal words
   ("should"/"must" → `section.description`; "researcher wrote it" → `answer.text`).
2. **Key distinctions** — explicit rules for the three pairs most likely to be confused:
   `section.description` vs `answer.text`, `question.text` vs `section.description`.
3. **Few-shot examples** — 10+ real DMP excerpts covering every label, drawn from the
   actual 10-sample corpus.
4. **Output constraint** — "You MUST output a JSON array with one entry for EVERY block
   in the TO CLASSIFY list — no explanation, no markdown."

---

## 12. Known Issues and Limitations

| Issue | Impact | Status |
|---|---|---|
| Duplicate character rendering in some PDFs | Blocks contain "HHeelllloo" patterns | Fixed in `text_cleaner._deduplicate_chars()` |
| pdf-direct blocks have no bounding boxes | Cannot render visual overlays for M07 | By design; use gold accuracy for eval |
| pdf-direct excluded from Section 7 (cross-model agreement) | Cannot compare block-by-block | Documented in notebook with note |
| OpenAI and Gemini providers configured but not used in experiments | No benchmark data | Providers ready; YAML configs not yet created |
| Whole-doc strategy requires large context window | Fails on very long DMPs | max_tokens=16384 in WholeDocStrategy |

---

## 13. Dependencies

**Core (always required):**
- `pdfplumber` ≥ 0.11 — PDF text and layout extraction
- `requests` ≥ 2.28 — HTTP client for Ollama
- `Pillow` ≥ 11.0 — optional page image export
- `python-dotenv` ≥ 1.0 — `.env` file support
- `pyyaml` ≥ 6.0 — experiment YAML configs

**Optional — notebooks:**
- `matplotlib`, `seaborn`, `pandas`

**Optional — REST server:**
- `fastapi`, `uvicorn`, `python-multipart`

**Provider SDKs (install separately):**
- `anthropic` — for Anthropic provider
- `openai` — for OpenAI provider
- `google-generativeai` — for Gemini provider

---

## 14. Reproduction Steps

```bash
# 1. Install
pip install -e ".[notebooks]"

# 2. Set API key (for Anthropic experiments)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Extract pdfplumber blocks for all 10 PDFs (notebook 001 does this)
#    Output → data/output/extracted/sampleN.json

# 4. Run any experiment
dmpbridge-experiment experiments/llama3.3-70b-batch.yaml
dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml --evaluate

# 5. Run pdf-direct experiment
dmpbridge-pdf --model claude-opus-4-8

# 6. Evaluate and compare in notebooks
jupyter lab notebooks/003_strategy_comparison.ipynb
```
