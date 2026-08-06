# Project rules and conventions

Facts about this project that are not obvious from the code, and that have cost
time when forgotten. Keep this file short — if something is discoverable by
reading the source, it does not belong here.

---

## Running Ollama on this machine

Start the server with **all** of these set. Environment variables only apply to a
freshly started process, so kill any running instance first
(`taskkill /F /IM ollama.exe`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OLLAMA_VULKAN=0 OLLAMA_SCHED_SPREAD=0 \
OLLAMA_KEEP_ALIVE=-1 LLAMA_ARG_FIT=off ollama serve
```

Each flag fixes a specific failure that has actually happened here:

| flag | without it |
|---|---|
| `OLLAMA_VULKAN=0` | Ollama picks the Vulkan backend and dies mid-run; Vulkan also ignores `CUDA_VISIBLE_DEVICES` and grabs different physical cards |
| `LLAMA_ARG_FIT=off` | `gemma4:e4b` crashes on load — `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)`. It is multimodal and Ollama 0.32.5 crashes auto-fitting its vision projector. The two llama models are text-only and unaffected, which makes this look like a GPU problem when it is not |
| `OLLAMA_SCHED_SPREAD=0` | small models get split across all four cards unnecessarily |

**Before any long run, check `ollama ps` reports `100% GPU`.** Anything less means
CPU offload and a multi-hour run instead of a 20-minute one. A stale
`llama-server.exe` holding VRAM is the usual cause — kill it and restart.

Only one large model fits comfortably at a time. `ollama stop <model>` before
loading another, or the next load fails on VRAM.

---

## Running the pipeline

```bash
dmpbridge-wholedoc --model <model> --extractor <extractor> --start 1 --end 10
```

- **There is no `--force` flag.** Stages 2–4 are cached by file existence, so a
  completed tag is skipped entirely. To re-run, delete the tag directory from
  `data/output/2_labeled/`, `3_structured/` and `4_final/`.
- **Do not delete `data/output/1_extracted/`.** It is keyed by *extractor*, not by
  model, and is shared by every model. Re-extracting is wasted work, and for
  `lighton` it is slow.
- `llama3.3:70b` takes ~112 s/sample, about 20 minutes for 10 documents. That
  exceeds a 10-minute foreground command limit — run it in the background.
- `gemma4:e4b` and `llama3.1:8b` take ~75 s for all 10.

---

## Interpreting results

- **The noise floor is ±0.002 F1 — one text block.** Measured 2026-08-06 by
  running one configuration three times: every count identical, F1 unmoved.
  `temperature: 0.0` is genuinely deterministic here. Differences above ~0.005
  are real signal. Earlier documents claiming ~3 points of run-to-run variation
  are wrong; that figure was assumed, never measured.
- **Always report both paths.** Path A scores stage 3 against the original
  annotation; Path B scores stage 4 (rules applied) against the revised
  annotation. They use different reference versions, so the gap between them
  measures the rules' contribution — it is not a second opinion on one number.
- Support differs between paths for the same reason. Compare scores, not counts.

---

## The prompt — `dmpbridge/prompts/system.py`

- **Treat it as whitespace-significant.** Moving three blank lines, with no
  wording change at all, moved F1 by 1.9 points and false positives by 13 on
  llama3.1:8b. Both layouts reproduced to within one block, so this is not noise.
  An editor that strips or adds a trailing blank line on save can change results
  more than most rewording. Do not reformat this file.
- **Prompt changes do not transfer between models.** One edit gained 0.004 on
  llama3.1:8b while costing 0.040 on gemma4:e4b and 0.006 on llama3.3:70b.
  **Re-run every model after any prompt change** — a partial re-run produces
  figures that cannot be compared.
- It is sent as the `system` field; the extractor blocks go separately as
  `prompt`. There is no template substitution, so a placeholder like
  `[Copy pdfplumber output here]` is sent to the model literally.

---

## Annotation rules — `data/input/Rules.xlsx`

**The column order is load-bearing.** It has changed once already, silently
remapping every row and dropping agreement with the reference files from 10/10 to
2/10 — without the action text changing, so nothing looked broken.

`RULE_FIELDS` in `dmpbridge/evaluation/annotation_rules.py` records the expected
order, and a test reads the sheet header at test time and fails on any mismatch.
If that test fails, the sheet was re-ordered — fix `RULE_FIELDS`, do not edit
the test.

---

## Notebooks and diagrams

- **The three results notebooks are generated.** Do not hand-edit them; edit the
  builder in `scripts/build/` and re-run, then execute with
  `jupyter nbconvert --to notebook --execute --inplace`.
- **Mermaid does not render inside `.ipynb` on GitHub**, only in `.md`. Diagrams
  in notebooks must be images in a cell output — that is what GitHub renders.
  `docs/pipeline.md` and `README.md` keep mermaid; the notebooks embed a
  matplotlib PNG from `scripts/build/make_pipeline_simple.py`.
- LaTeX (`$$…$$`, `$F_1$`) does not render reliably in notebook viewers either.
  Use markdown tables and fenced code blocks.
- Two diagram sources describe the same pipeline. **Changing one means changing
  the other.**

---

## Repository

- `Report-doc/*.png` and `*.docx` are build artefacts, reproducible from
  `scripts/build/`, and are gitignored. The worklogs in `Report-doc/worklog/` are
  **not** artefacts — they hold reasoning git history cannot reconstruct, and
  stay tracked.
- `tests/` is gitignored, so a new test file needs `git add -f`. The existing
  suite is tracked.
- Notebooks are marked `linguist-documentation` in `.gitattributes`; their stored
  chart images otherwise make GitHub report this as a Jupyter project rather than
  a Python one.
- Commit messages should describe the change. Several in the history say things
  like "improve error handling" on commits that touch neither.
