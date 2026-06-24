"""
Run pipeline --no-smooth for both models across all 10 samples.
Saves to data/llmlabeled/<sample>_<model>-nosmooth.json
"""
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
from dmpbridge.pipeline import process_pdf

PDF_DIR    = Path("data/pdfsamples")
OUT_DIR    = Path("data/llmlabeled")
MODELS     = ["llama3.3:70b", "llama3.1:8b"]
SAMPLES    = sorted(PDF_DIR.glob("sample*.pdf"),
                    key=lambda p: int(p.stem.replace("sample", "")))

for model in MODELS:
    model_tag = model.replace(":", "-")
    print(f"\n{'='*60}")
    print(f"  Model: {model}  (no-smooth)")
    print(f"{'='*60}")
    for pdf in SAMPLES:
        stem   = pdf.stem                          # e.g. "sample1"
        out    = OUT_DIR / f"{stem}_{model_tag}-nosmooth.json"
        struct = OUT_DIR / f"{stem}_{model_tag}-nosmooth_structured.json"
        if out.exists():
            print(f"  SKIP  {out.name}  (already exists)")
            continue
        t0 = time.time()
        print(f"  RUN   {pdf.name} → {out.name} ...", end=" ", flush=True)
        try:
            process_pdf(
                pdf,
                model=model,
                output=out,
                structured_output=struct,
                raw_dir=None,
                smooth=False,
            )
            print(f"done ({time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"ERROR: {exc}")

print("\nAll done.")
