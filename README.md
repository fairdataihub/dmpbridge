# DMPBridge

Turn Data Management Plan PDFs into structured, machine-readable records — using a local
LLM, with nothing leaving your machine.

A DMP is a document researchers write to describe what data a project will produce and how
it will be stored and shared. They arrive as PDFs, which makes them hard to search or
compare at scale. DMPBridge reads one, works out what each piece of text *is* — a section
heading, a question, an answer — and outputs structured JSON.

> **This is an active research project — things change often.** 
---

## How it works

```mermaid
flowchart TD
    PDF["<b>DMP PDF</b>"]

    PDF --> READ["<b>Read the PDF</b><br/><small>pdfplumber</small>"]
    READ --> S1["<b>1. Text blocks</b>"]
    S1 --> LABEL["<b>Label each block</b><br/><small>llama3.1:8b · gemma4:e4b · llama3.3:70b</small>"]
    LABEL --> S2["<b>2. Labeled blocks</b>"]
    S2 --> BUILD["<b>Build the structure</b>"]
    BUILD --> S3["<b>3. Structured JSON</b>"]

    S3 --> PATHA["<b>Path A</b><br/>score vs old annotation"]
    S3 --> RULES["<b>Apply the rules</b><br/><small>Rules.xlsx</small>"]
    RULES --> S4["<b>4. Final JSON</b>"]
    S4 --> PATHB["<b>Path B</b><br/>score vs new annotation"]

    classDef input  fill:#1E406E,stroke:#1E406E,color:#ffffff
    classDef step   fill:#F4F7FB,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef cached fill:#ffffff,stroke:#0F766E,stroke-width:2px,color:#111
    classDef data   fill:#ffffff,stroke:#1E406E,stroke-width:2px,color:#111
    classDef rules  fill:#ffffff,stroke:#B45309,stroke-width:2px,color:#111
    classDef pathA  fill:#EDF3FA,stroke:#3C6FA8,stroke-width:2px,color:#111
    classDef pathB  fill:#FDF4E9,stroke:#B45309,stroke-width:2px,color:#111

    class PDF input
    class READ,LABEL,BUILD,RULES step
    class S1 cached
    class S2,S3 data
    class S4 rules
    class PATHA pathA
    class PATHB pathB
```

Each numbered box is written to disk, so any stage can be inspected on its own.
More detail in **[docs/pipeline.md](docs/pipeline.md)**.

Every block gets one of five labels:

| Label | Meaning |
|---|---|
| `title` | Document title |
| `section.title` | Section heading |
| `section.description` | Instructions written by the funder |
| `question.text` | A question or prompt |
| `answer.text` | The researcher's response |

---

## Where the project stands

Research code — results are provisional and the evaluation set is small.

> **Scores pending re-evaluation.** pdfplumber's extraction+labeling was rewritten to a
> whole-document, visual-signal-marker approach (see [Choosing an extractor](#choosing-an-extractor))
> after the last F1 numbers were measured, so the old table is no longer accurate for the
> current code and has been removed rather than shown stale. Re-run
> `dmpbridge-experiment experiments/<model>-wholedoc.yaml --evaluate` per model to get
> current numbers.

`pdfplumber` is the only extractor in the pipeline now — Docling and LightOnOCR have both
been removed. That's 3 planned (one per model) configurations, 0 currently re-evaluated
under the current pdfplumber implementation.

Run-to-run noise, last measured against the previous pdfplumber implementation, was
±0.002 F1 (one configuration run three times, every count identical) — not the ~3-point
figure an earlier version of this file assumed but never measured. Differences smaller
than ~0.005 are not meaningful.

---

## Quick start

**Requirements:** Python 3.10+ · [Ollama](https://ollama.com) running locally

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS / Linux

pip install -e .
ollama pull gemma4:e4b           # ~3 GB — recommended
```

Label one PDF:

```python
import dmpbridge

blocks = dmpbridge.process_pdf(
    "document.pdf",
    model="gemma4:e4b",
    extractor="pdfplumber",
    structured_output="structured.json",
)
```

Or the whole sample set:

```bash
dmpbridge-wholedoc --model gemma4:e4b --extractor pdfplumber --start 1 --end 10
```

Re-runs skip samples that already have output.

**Or the smallest possible example** — edit [`demo/config.yaml`](demo/config.yaml) (just
a model, extractor, and sample range) and run:

```bash
python scripts/run_demo.py
```

which writes each stage's result into `demo/output/{labeled,structured,final}/`. The same
config also drives [`notebooks/demo-from-yaml-config.ipynb`](notebooks/demo-from-yaml-config.ipynb),
which shows the YAML as input and the final document as output side by side.

---

## How extraction works

`pdfplumber` is the only extractor implemented — no GPU needed, no extra install. It reads
the whole document at once and sends it to the model in a single call. Words visually
emphasized relative to the document's own body-text font (bold, or a larger size) are
wrapped in `**markers**`, and italic words in `_markers_`, so the model gets the PDF's own
visual structure as a signal without needing bounding-box or font-size fields — extraction
and labeling are fused into one step rather than separate segment-then-classify calls.

It assumes a text layer exists (not a scanned/image-only PDF). Docling and LightOnOCR —
an OCR-capable alternative for scanned documents — were both tried in this project and
removed. If a scanned-PDF use case comes up, that capability would need to be rebuilt.

---

## Where the output goes

```
data/output/
├── 1_extracted/<extractor>/sampleN.json     text blocks, no labels
├── 2_labeled/<tag>/sampleN.json             the same blocks, labeled
├── 3_structured/<tag>/sampleN.json          nested into the DMP Tool schema
└── 4_final/<tag>/sampleN.json               annotation rules applied
```

`<tag>` is `<model>_<extractor>_whole_doc`. Filenames are the same at every stage, so you
can open `sampleN.json` in each folder and follow one document through.

**Stage 1 is keyed by extractor, not by model**, and is cached — reading a PDF doesn't
depend on which LLM labels it. Labeling with three models costs one read, not three:

```bash
dmpbridge-wholedoc --model llama3.1:8b   # reads and caches
dmpbridge-wholedoc --model gemma4:e4b    # reuses the cache

dmpbridge-wholedoc --model gemma4:e4b --no-cache   # force re-read
dmpbridge-wholedoc --model gemma4:e4b --no-rules   # skip stage 4
```

To read PDFs with no LLM involved at all:

```bash
python scripts/extract_pdfplumber.py
```

---

## Scoring

Output is compared against hand-annotated reference documents. The annotation standard
changed partway through the project, so everything is scored twice:

- **Path A** — the structured output against the original annotation
- **Path B** — after filling blank questions using `data/input/Rules.xlsx`, against the newer one

Both use the same scoring: a predicted item matches a reference item when enough of its
words are contained in it (**100% by default** — every word, no partial credit; pass
`threshold=0.75` to relax it), then precision, recall and F1 as usual.

```bash
dmpbridge-evaluate      gemma4-e4b_pdfplumber_whole_doc    # Path A
dmpbridge-evaluate-new  gemma4-e4b_pdfplumber_whole_doc    # Path B
```

Both paths are also driven from one YAML file, which is what the numbers table above is
built from:

```bash
dmpbridge-experiment experiments/llama3.1-8b-wholedoc.yaml --evaluate
```

A new annotation source ("Path C") doesn't need new code — it's a new entry in an
`evaluation:` list in the YAML (`experiments/full-example.yaml` shows every field). See
`EvaluationPath` in `dmpbridge/evaluation/evaluate.py`.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Running on multiple GPUs

Ollama may pick the Vulkan backend over CUDA, which is unstable across several cards and
ignores `CUDA_VISIBLE_DEVICES`. **`gemma4:e4b`** — the recommended model above — also
crashes on load under recent Ollama versions unless one more flag is set, since it's
multimodal and the crash is in fitting its vision projector, not a GPU problem. Start the
server with all of these, killing any running instance first (env vars only apply to a
freshly started process):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OLLAMA_VULKAN=0 OLLAMA_SCHED_SPREAD=0 \
OLLAMA_KEEP_ALIVE=-1 LLAMA_ARG_FIT=off ollama serve
```

Then check `ollama ps` reports **100% GPU**. Anything less means part of the model spilled
to CPU, and a large model will take hours instead of minutes. 
