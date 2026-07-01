"""Run whole-document Claude classification on all 10 samples.

Output files: data/llmlabeled/sampleN_claude-opus-4-8_whole_doc.json
"""
import sys, json
sys.path.insert(0, ".")

import anthropic
from pathlib import Path
from dmpbridge.extractor import extract_blocks
from dmpbridge.classifier import LABELS, SYSTEM_PROMPT
from dmpbridge import config

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
MODEL   = "claude-opus-4-8"
PDF_DIR = Path("data/pdfsamples")
OUT_DIR = Path("data/llmlabeled")

for i in range(1, 11):
    pdf_path = PDF_DIR / f"sample{i}.pdf"
    out_path = OUT_DIR / f"sample{i}_claude-opus-4-8_whole_doc.json"

    if out_path.exists():
        print(f"[sample{i}] already exists — skipping")
        continue

    print(f"\n[sample{i}] extracting blocks from {pdf_path.name} ...")
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
    response = client.messages.create(
        model=MODEL,
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text if response.content else ""
    print(f"[sample{i}] in={response.usage.input_tokens:,}  out={response.usage.output_tokens:,}  chars={len(raw):,}")

    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        raw_clean = raw_clean.split("```", 2)[1]
        if raw_clean.startswith("json"):
            raw_clean = raw_clean[4:]
        raw_clean = raw_clean.strip()

    try:
        parsed = json.loads(raw_clean)
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
