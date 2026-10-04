"""Run llama3.3:70b on a saved run's prompt with part of the model on the CPU.

The notebook skips a model that is not 100% on the GPU, because CPU offload can
turn minutes into hours. With one of the three GPUs lost, the 70B does not fit, so
this script runs it anyway - a measured number of layers on the two healthy GPUs
and the rest on the CPU - times it, and saves the result into the run's folder so
the notebook can load and evaluate it (RUN_NAME = "<run>").

    python scripts/run_70b_partial_offload.py v2 [--layers 62]
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from dmpbridge.extractors import get_extractor

MODEL, HOST, NUM_CTX, NUM_PREDICT, SAMPLE = "llama3.3:70b", "http://localhost:11434", 32768, 8000, 14
TOTAL_LAYERS = 81

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--layers", type=int, default=62, help="layers on the GPU (of 81); the rest run on the CPU")
args = ap.parse_args()

run_dir = Path("data/output/rda/runs") / args.run
record = (run_dir / "prompt.txt").read_text(encoding="utf-8")
marker = "\n\nPROMPT (the schema and the document text go in the braces):\n"
system = record[len("SYSTEM:\n"):record.index(marker)]
template = record[record.index(marker) + len(marker):]

schema = json.loads(Path("data/output/rda/maDMP-schema-1.2.json").read_text(encoding="utf-8"))
defs = schema["$defs"]


def resolve_refs(node):
    if isinstance(node, dict):
        if "$ref" in node:
            return resolve_refs(defs[node["$ref"].split("/")[-1]])
        return {k: resolve_refs(v) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [resolve_refs(v) for v in node]
    return node


schema_full = resolve_refs(schema)
schema_text = json.dumps(schema_full, separators=(",", ":"))
dmp_text = get_extractor("pdfplumber").extract(Path("data/input/pdfs") / f"sample{SAMPLE}.pdf")[0]["text"]
prompt = template.replace("{schema_text}", schema_text).replace("{dmp_text}", dmp_text)

for other in ("llama3.1:8b", "gemma4:e4b"):
    subprocess.run(["ollama", "stop", other], check=False)

options = {"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "num_gpu": args.layers}
t0 = time.perf_counter()
r = requests.post(f"{HOST}/api/generate", timeout=600,
                  json={"model": MODEL, "keep_alive": -1, "options": options})
load = time.perf_counter() - t0
if r.status_code != 200:
    raise SystemExit(f"load failed after {load:.0f} s: {r.text[:400]}")
placement = next((l for l in subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()
                  if l.startswith(MODEL)), "")
print(f"loaded in {load:.0f} s with {args.layers}/{TOTAL_LAYERS} layers on the GPU")
print("ollama ps:", " ".join(placement.split()), flush=True)

t0 = time.perf_counter()
r = requests.post(f"{HOST}/api/generate", timeout=4 * 3600,
                  json={"model": MODEL, "system": system, "prompt": prompt, "stream": False,
                        "format": schema_full, "keep_alive": -1, "options": options})
elapsed = time.perf_counter() - t0
r.raise_for_status()
body = r.json()
print(f"{MODEL}: {elapsed:.0f} s   tokens sent: {body.get('prompt_eval_count', 0):,}   "
      f"generated: {body.get('eval_count', 0):,}   stopped because: {body.get('done_reason')}", flush=True)

tag = MODEL.replace(":", "-")
try:
    result = json.loads(body["response"])
except json.JSONDecodeError as e:
    (run_dir / f"sample{SAMPLE}.rda.{tag}.raw.txt").write_text(body["response"], encoding="utf-8")
    raise SystemExit(f"no valid JSON ({e.msg} at character {e.pos:,}); raw text saved")
out = run_dir / f"sample{SAMPLE}.rda.{tag}.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"saved -> {out}")

info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
info["timings"][f"{MODEL} sample{SAMPLE}"] = {
    "seconds": round(elapsed), "model_load_seconds": round(load),
    "tokens_sent": body.get("prompt_eval_count"), "tokens_generated": body.get("eval_count"),
    "note": f"run by scripts/run_70b_partial_offload.py on {datetime.now():%Y-%m-%d %H:%M} with "
            f"{args.layers}/{TOTAL_LAYERS} layers on the GPU ({' '.join(placement.split())})"}
info["skipped"] = [s for s in info.get("skipped", []) if not s.startswith(MODEL)]
(run_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print("run.json updated")
