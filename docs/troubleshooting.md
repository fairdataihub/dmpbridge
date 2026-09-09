# Troubleshooting

[← Main README](../README.md) · [All docs](README.md)

Failures that have actually happened on this project, and what fixes each one. **Find your
symptom in the table, then jump to it.**

| What you're seeing | Jump to |
|---|---|
| `could not connect to ollama` | Ollama isn't running — start it with `ollama serve` and re-run |
| `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)` — `gemma4:e4b` won't load | [Ollama environment variables](#ollama-multiple-gpus-and-the-gemma4e4b-load-crash) |
| Ollama grabs the wrong GPUs, or dies mid-run | [Ollama environment variables](#ollama-multiple-gpus-and-the-gemma4e4b-load-crash) |
| a run is taking hours instead of minutes | [Check `100% GPU`](#check-100-gpu-before-a-long-run) |
| `operator torchvision::nms does not exist`, or `Could not import module 'AutoProcessor'` | [CPU-only torch on Windows](#lightonocr-cpu-only-torch-on-windows) |
| LightOnOCR produces garbage, or the fallback does nothing | [CPU-only torch on Windows](#lightonocr-cpu-only-torch-on-windows) |
| "already exists — skipping", nothing runs | [A document was skipped](#a-document-was-skipped) |
| the command times out before finishing | [A run times out](#a-run-times-out-in-the-foreground) |
| output reads fine but doesn't match the PDF | [Output is fluent but wrong](#output-is-fluent-but-wrong) |

---

## Ollama: multiple GPUs, and the `gemma4:e4b` load crash

Ollama may pick the Vulkan backend over CUDA, which is unstable across several cards and
ignores `CUDA_VISIBLE_DEVICES` entirely — it grabs different physical cards than the ones
you asked for. Separately, **`gemma4:e4b` — the recommended model — crashes on load** under
recent Ollama versions with `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)`. That one
is not a GPU problem: gemma4 is multimodal and the crash is in auto-fitting its vision
projector. The text-only llama models are unaffected, which makes it look like a GPU fault
when it isn't.

Environment variables only apply to a freshly started process, so **kill any running
instance first** (`taskkill /F /IM ollama.exe` on Windows), then start the server with all
of these:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OLLAMA_VULKAN=0 OLLAMA_SCHED_SPREAD=0 \
OLLAMA_KEEP_ALIVE=-1 LLAMA_ARG_FIT=off ollama serve
```

| Flag | Without it |
|---|---|
| `OLLAMA_VULKAN=0` | Ollama picks the Vulkan backend and dies mid-run, and ignores `CUDA_VISIBLE_DEVICES` |
| `LLAMA_ARG_FIT=off` | `gemma4:e4b` crashes on load while auto-fitting its vision projector |
| `OLLAMA_SCHED_SPREAD=0` | small models get split across all four cards unnecessarily |
| `OLLAMA_KEEP_ALIVE=-1` | the model unloads between samples and every document pays the load cost again |

---

## Check `100% GPU` before a long run

```bash
ollama ps
```

Anything less than `100% GPU` means part of the model spilled to CPU, and a large model
takes hours instead of minutes. A stale `llama-server.exe` still holding VRAM is the usual
cause — kill it and restart the server.

Only one large model fits comfortably at a time. Run `ollama stop <model>` before loading
another, or the next load fails on VRAM.

---

## LightOnOCR: CPU-only torch on Windows

`pip install -e ".[lighton]"` on Windows pulls the **CPU-only** torch build, with which
LightOnOCR cannot run. The failure is quiet: `--fallback auto` fails soft and the run
proceeds with unusable text.

On a machine with a CUDA GPU, install the CUDA build explicitly:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Check it took:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

It must print `True`.

torch and torchvision must be **matching builds**. A torchvision compiled against a
different torch fails at import with `operator torchvision::nms does not exist`, which
surfaces as a misleading `Could not import module 'AutoProcessor'` from transformers.

---

## A document was skipped

Stages 2–4 are cached by file existence, so a document that already has output is skipped
entirely. **There is no `--force` flag.** To re-run one, delete its files from
`data/output/2_labeled/<tag>/`, `3_structured/<tag>/` and `4_final/<tag>/`.

Do **not** delete `data/output/1_extracted/`. It is keyed by extractor, not by model, and is
shared by every model — re-extracting is wasted work. Use `--no-cache` if you genuinely need
a fresh read.

`python scripts/rerun.py --all --dry-run` shows what a re-run would delete before it does
it.

---

## A run times out in the foreground

`llama3.3:70b` takes about 112 s per document — roughly 20 minutes for 10 documents, which
exceeds a 10-minute foreground command limit. Run it in the background.

`gemma4:e4b` and `llama3.1:8b` do all 10 in about 75 seconds.

---

## Output is fluent but wrong

If a document's structured output reads as plausible prose that doesn't match the PDF, the
text layer is probably broken and the model hallucinated from garbage. Check the log for the
garbled-text warning and see [extraction.md](extraction.md#broken-or-scanned-pdfs).
