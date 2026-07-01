"""Run whole-document Ollama classification on all 10 samples.

Usage:
    python run_wholedoc_ollama.py --model llama3.3:70b
    python run_wholedoc_ollama.py --model llama3.1:8b

Output files: data/llmlabeled/sampleN_{model}_whole_doc.json
  e.g.  sample1_llama3.3-70b_whole_doc.json
"""
import argparse
import json
import sys
sys.path.insert(0, ".")

from pathlib import Path

import requests

from dmpbridge import config
from dmpbridge.classifier import LABELS, SYSTEM_PROMPT, _OUTPUT_SCHEMA
from dmpbridge.extractor import extract_blocks

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama3.3:70b",
                    help="Ollama model name (e.g. llama3.3:70b, llama3.1:8b)")
parser.add_argument("--host", default=config.HOST)
args = parser.parse_args()

MODEL   = args.model
HOST    = args.host.rstrip("/")
# file tag: llama3.3:70b -> llama3.3-70b
TAG     = MODEL.replace(":", "-").replace(".", ".")
PDF_DIR = Path("data/pdfsamples")
OUT_DIR = Path("data/llmlabeled")

print(f"Model : {MODEL}")
print(f"Host  : {HOST}")
print(f"Tag   : {TAG}-wholedoc")
print()

for i in range(1, 11):
    pdf_path = PDF_DIR / f"sample{i}.pdf"
    out_path = OUT_DIR / f"sample{i}_{TAG}_whole_doc.json"

    if out_path.exists():
        print(f"[sample{i}] already exists — skipping")
        continue

    print(f"[sample{i}] extracting blocks ...")
    blocks = extract_blocks(pdf_path)

    payload = [
        {
            "id":     j,
            "text":   b["text"],
            "bold":   b["is_bold"],
            "italic": b.get("is_italic", False),
            "page":   b["page"],
        }
        for j, b in enumerate(blocks)
    ]

    prompt = (
        f"CLASSIFY ALL BLOCKS — return a JSON array with exactly {len(payload)} entries, "
        f"one per block, in the same order:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    print(f"[sample{i}] sending {len(payload)} blocks to {MODEL} ...")
    try:
        resp = requests.post(
            f"{HOST}/api/generate",
            json={
                "model":   MODEL,
                "system":  SYSTEM_PROMPT,
                "prompt":  prompt,
                "stream":  False,
                "format":  _OUTPUT_SCHEMA,
                "options": {"temperature": 0.0, "num_ctx": 32768},
            },
            timeout=600,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[sample{i}] ERROR: {e}")
        continue

    raw = resp.json().get("response", "")
    print(f"[sample{i}] response: {len(raw):,} chars")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    except json.JSONDecodeError as e:
        print(f"[sample{i}] JSON parse error: {e}")
        parsed = []

    print(f"[sample{i}] parsed {len(parsed)} entries (expected {len(blocks)})")

    result = [dict(b) for b in blocks]
    for entry in parsed:
        idx = entry.get("id")
        lbl = entry.get("label", "answer.text")
        if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
            result[idx]["label"] = lbl

    for b in result:
        if not b.get("label"):
            b["label"] = "answer.text"

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sample{i}] saved to {out_path}")

print("\nAll done.")
