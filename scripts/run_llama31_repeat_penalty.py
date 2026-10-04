"""Run llama3.1:8b alone on a saved run's prompt with a wider repeat-penalty window.

With the default settings (penalty window 64 tokens) llama3.1:8b loops on prompts
that ask for every dataset: it writes the same dataset block until the token cap and
never closes the JSON. This script keeps the run's prompt exactly as saved and only
widens the window the repeat penalty looks at, trying a few strengths until the
answer is valid JSON. The result is saved into the run's folder with a note in
run.json saying which setting produced it.

    python scripts/run_llama31_repeat_penalty.py v3
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from dmpbridge.extractors import get_extractor

MODEL, HOST, NUM_CTX, NUM_PREDICT, SAMPLE = "llama3.1:8b", "http://localhost:11434", 32768, 8000, 14
TRIES = [(512, 1.1), (512, 1.2), (1024, 1.3)]        # (repeat_last_n, repeat_penalty)

ap = argparse.ArgumentParser()
ap.add_argument("run")
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

for other in ("gemma4:e4b", "llama3.3:70b"):
    subprocess.run(["ollama", "stop", other], check=False, capture_output=True)

tag = MODEL.replace(":", "-")
for last_n, penalty in TRIES:
    options = {"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
               "repeat_last_n": last_n, "repeat_penalty": penalty}
    t0 = time.perf_counter()
    r = requests.post(f"{HOST}/api/generate", timeout=3600,
                      json={"model": MODEL, "system": system, "prompt": prompt, "stream": False,
                            "format": schema_full, "keep_alive": -1, "options": options})
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    line = (f"repeat_last_n={last_n} repeat_penalty={penalty}: {elapsed:.0f} s, "
            f"{body.get('eval_count', 0):,} tokens generated, stopped because: {body.get('done_reason')}")
    try:
        result = json.loads(body["response"])
    except json.JSONDecodeError as e:
        print(f"{line} -> still no valid JSON ({e.msg} at {e.pos:,})", flush=True)
        continue
    n_ds = len(result.get("dmp", {}).get("dataset") or [])
    print(f"{line} -> VALID JSON, {n_ds} datasets", flush=True)
    out = run_dir / f"sample{SAMPLE}.rda.{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    info["timings"][f"{MODEL} sample{SAMPLE}"] = {
        "seconds": round(elapsed), "model_load_seconds": 0,
        "tokens_sent": body.get("prompt_eval_count"), "tokens_generated": body.get("eval_count"),
        "note": f"run by scripts/run_llama31_repeat_penalty.py on {datetime.now():%Y-%m-%d %H:%M}: same prompt, "
                f"but repeat_last_n={last_n} and repeat_penalty={penalty} instead of the defaults (64, 1.1), "
                f"because with the defaults the model looped until the token cap (see the .raw.txt files)"}
    info["skipped"] = [s for s in info.get("skipped", []) if not s.startswith(MODEL)]
    (run_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("run.json updated")
    break
else:
    raise SystemExit("no setting produced valid JSON; nothing saved")
