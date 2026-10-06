"""Run saved RDA prompt versions on a set of samples, one after another.

For each version: the system prompt is taken verbatim from a saved prompt.txt, put
into the extraction notebook's builder, and the notebook is executed with that
version's RUN_NAME and the given samples (results already saved in the run folder are
loaded, not regenerated). Then the overview notebook is executed for the version, which
saves its overview and per-sample figures into the run folder. At the end both builders
are set back to the version given with --restore.

The script starts its own Ollama server (with the flags this machine needs) and stops
it at the end, so a long series does not depend on a server started elsewhere.

    python scripts/run_rda_versions.py v1=v1-sample14 v2=v2-sample14 --samples 1-10 --restore v7
      (v1=v1-sample14: run name v1, prompt taken from runs/v1-sample14/prompt.txt)
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

RUNS = Path("data/output/rda/runs")
EXTRACT = Path("scripts/build/build_rda_extraction_notebook.py")
OVERVIEW = Path("scripts/build/build_rda_overview_notebook.py")
JUPYTER = Path("venv/Scripts/jupyter.exe")
MARKER = "\n\nPROMPT (the schema and the document text go in the braces):\n"
OLLAMA_ENV = {"CUDA_VISIBLE_DEVICES": "0,1,2", "OLLAMA_VULKAN": "0", "OLLAMA_SCHED_SPREAD": "0",
              "OLLAMA_KEEP_ALIVE": "-1", "LLAMA_ARG_FIT": "off"}

ap = argparse.ArgumentParser()
ap.add_argument("versions", nargs="+", help="RUN_NAME=prompt-folder, or just RUN_NAME if its folder holds the prompt")
ap.add_argument("--samples", default="1-10")
ap.add_argument("--restore", required=True, help="run folder whose prompt and name the builders get back at the end")
args = ap.parse_args()
lo, _, hi = args.samples.partition("-")
samples = list(range(int(lo), int(hi or lo) + 1))


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def system_of(folder):
    t = (RUNS / folder / "prompt.txt").read_text(encoding="utf-8")
    return t[len("SYSTEM:\n"):t.index(MARKER)]


def set_extraction(run_name, system, sample_list):
    t = EXTRACT.read_text(encoding="utf-8")
    start = t.index('SYSTEM = """') + len('SYSTEM = """')
    t = t[:start] + system + t[t.index('"""', start):]
    t = re.sub(r'^SAMPLES  = \[.*\]$', f"SAMPLES  = {sample_list}", t, count=1, flags=re.M)
    t = re.sub(r'^RUN_NAME = .*$', f'RUN_NAME = "{run_name}"', t, count=1, flags=re.M)
    EXTRACT.write_text(t, encoding="utf-8")
    subprocess.run([sys.executable, str(EXTRACT)], check=True)


def set_overview(run_name):
    t = OVERVIEW.read_text(encoding="utf-8")
    t = re.sub(r'^RUN    = ".*?"', f'RUN    = "{run_name}"', t, count=1, flags=re.M)
    OVERVIEW.write_text(t, encoding="utf-8")
    subprocess.run([sys.executable, str(OVERVIEW)], check=True)


def execute(notebook):
    r = subprocess.run([str(JUPYTER), "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=-1", notebook], capture_output=True, text=True)
    if r.returncode:
        log(f"  {notebook} FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
    return r.returncode == 0


# ── start a fresh Ollama server ───────────────────────────────────────────────
for exe in ("ollama app.exe", "ollama.exe", "llama-server.exe"):
    subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)
time.sleep(2)
server = subprocess.Popen(["ollama", "serve"], env={**os.environ, **OLLAMA_ENV},
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        version = requests.get("http://localhost:11434/api/version", timeout=2).json()["version"]
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Ollama did not start")
log(f"Ollama {version} started; samples {samples}")

try:
    for spec in args.versions:
        run_name, _, folder = spec.partition("=")
        folder = folder or run_name
        t0 = time.time()
        log(f"{run_name}: prompt from runs/{folder}/prompt.txt")
        set_extraction(run_name, system_of(folder), samples)
        ok = execute("notebooks/pdf-to-rda-dmp-json.ipynb")
        got = sorted(f.name for f in (RUNS / run_name).glob("sample*.rda.*.json")) if (RUNS / run_name).exists() else []
        failed = sorted(f.name for f in (RUNS / run_name).glob("sample*.rda.*.raw.txt")) if (RUNS / run_name).exists() else []
        log(f"{run_name}: notebook {'ok' if ok else 'FAILED'}, {len(got)} JSON results, {len(failed)} failed "
            f"({', '.join(failed) or 'none'}), {(time.time() - t0) / 60:.0f} min")
        if ok:
            set_overview(run_name)
            execute("notebooks/rda-evaluation-overview.ipynb")
            log(f"{run_name}: overview figures saved")
finally:
    set_extraction(args.restore, system_of(args.restore), samples)
    set_overview(args.restore)
    execute("notebooks/rda-evaluation-overview.ipynb")
    log(f"builders set back to {args.restore}")
    server.terminate()
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    log("Ollama stopped")
