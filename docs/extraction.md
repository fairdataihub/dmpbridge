# Extraction

How a PDF becomes stage-1 text blocks, and what to do when a PDF won't read.

Stage 1 is keyed by **extractor**, not by model, and is cached in
`data/output/1_extracted/<extractor>/sampleN.json`. Labeling the same documents with four
models costs one read, not four.

---

## The three extractors

### `pdfplumber` (default)

No GPU needed, no extra install. It reads the whole document at once and sends it to the
model in a single call.

Words visually emphasized relative to the document's own body-text font (bold, larger, or
underlined) are wrapped in `**bold**`, `_italic_`, or `++underline++` markers, so the model
gets the PDF's own visual structure as a signal without needing bounding-box or font-size
fields — extraction and labeling are fused into one step rather than separate
segment-then-classify calls.

It assumes a text layer exists, so it cannot read a scanned or image-only PDF.

### `lightonocr` (alternative)

A vision-LLM (LightOnOCR-2-1B) that reads each page as an image instead, using the same
marker convention. Needs a CUDA GPU and `pip install -e ".[lighton]"`.

Scores lower and runs slower than pdfplumber on this project's documents so far, but works
on scanned or image-only PDFs pdfplumber can't read at all — see
`notebooks/comparison-gemma-pdfplumber-vs-lightonocr.ipynb`.

**On Windows, plain pip installs the CPU-only torch build, with which LightOnOCR cannot
run** — and the `--fallback auto` rescue then fails soft and the run proceeds with unusable
text. This was observed directly: two runs from a venv with CPU torch produced garbage
output while the same command worked from an environment with CUDA torch. See
[troubleshooting.md](troubleshooting.md) for the CUDA install.

### `docling` (alternative)

Docling's layout model reads each page, and the text is built from Docling's *native* page
cells rather than its Markdown export (which drops every font): pdfplumber's bold/italic
rules applied to each word's font name and size, hyperlink rectangles as `++underline++`,
and Docling's own heading label where the font marks nothing.

On sample 2 it reproduces pdfplumber's markers exactly (42 bold, 41 italic, 11 underlined)
and matches them on 8 of 10 documents. With `gemma4:e4b` it scores 0.924 Path A / 0.910
Path B against pdfplumber's 0.946 / 0.951: ahead on samples 2 and 5, level on seven, and
behind only on sample 6, whose headings are drawn underlines with no link behind them —
Docling has no shape data for those at any level.

Runs on CPU in 0.1–3 s per document; needs `pip install -e ".[docling]"`.

All three extractors side by side — markers per document, both scoring paths, per class,
per document, confusion matrices — are in `notebooks/comparison-gemma-extractors.ipynb`.

---

## Native dumps

`sampleN.native.json` is the extractor's raw reading of the PDF *before* any marker rule.
For pdfplumber that is every word with its font name, size and box, the drawn rectangles
and lines that underline detection works from, and hyperlinks with their URIs, written to
`1_extracted/pdfplumber/`. It never changes the stage-1 text; it is there so a marker can
be traced back to what produced it.

`dmpbridge-wholedoc` writes it **by default** for pdfplumber and docling runs, for every
sample in range that lacks one (cached or already labeled). Pass `--no-save-native` to skip
it, or produce dumps for all samples without touching the model stages:

```bash
python scripts/native_dump.py --extractor pdfplumber   # or docling
```

Docling also leaves `sampleN.md` next to it — Docling's own Markdown export — plus a fuller
native result with layout clusters, confidences and page images.

To read PDFs with no LLM involved at all:

```bash
python scripts/extract_pdfplumber.py
```

---

## Broken or scanned PDFs

A PDF can have a text layer whose fonts carry no character-to-text mapping. Extraction then
*succeeds* as garbage — `(cid:NN)` tokens or mojibake — and the model hallucinates a fluent
document from it. This was observed on sample 11.

The pipeline checks every stage-1 text and logs a loud warning when it does not look like
readable language. **The warning is a guard, not a gate: by default the run still
proceeds**, so check the log for `[fallback]` lines before trusting output for a new
document.

### Automatic rescue — `--fallback auto`

```bash
dmpbridge-wholedoc --model gemma4:e4b --extractor pdfplumber --fallback auto --start 1 --end 13
```

A document whose text fails the check is re-extracted with LightOnOCR and labeled from that
text instead — per document, opt-in, clean documents untouched. You'll see two `[fallback]`
lines in the terminal saying why and what was used.

The primary extractor's stage-1 cache keeps what it really read (the garbage), while the
accepted rescue text is cached under the **fallback's own** stage-1 directory
(`1_extracted/lightonocr/sampleN.json`), so the rescue is inspectable on disk and later runs
reuse it — making rescued documents reproducible instead of re-rolling the OCR each time.

The Python API takes the same option — `process_pdf(..., fallback="auto")`, or a name, or
an ordered list — so an application embedding the package gets the same behaviour without
the CLI.

**If the rescue itself cannot run** (for example CPU-only torch), the run currently proceeds
with a loud warning and unusable output.

### Manual options

1. **`--extractor lightonocr`** — reads the page image, unaffected by the text layer, and it
   structures OCR'd documents far better than text OCR does. On the one broken PDF measured
   it produced proper sections, questions and answers where docling-OCR text collapsed into
   a single fused question. Needs the CUDA GPU.
2. **`--extractor docling --force-ocr`** — *experimental, not part of the production
   fallback.* OCRs every page (`OcrMode.FULL_PAGE`) instead of trusting the text layer.
   CPU-only, so it works without a GPU, but bold/italic markers are mostly lost with the
   fonts and structure suffers — on the one broken PDF measured it fused five sections into
   one question. The stage-1 cache is keyed by extractor only, so clear the cached
   `sampleN.json` or pass `--no-cache`, or the garbage text is reused.
3. **`pdfplumber` has no fallback** — it can only read the text layer.
