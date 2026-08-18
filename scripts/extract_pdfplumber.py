"""Extract text from the sample PDFs with pdfplumber — no LLM, no labeling.

Writes one JSON file per sample containing the extractor's raw output. For
pdfplumber this is a single ``{"text": "..."}`` entry — the whole document
as one string with **bold**/_italic_ visual-signal markers — since
pdfplumber fuses extraction and labeling into one whole-document model
call rather than producing separate per-line blocks (see
PdfplumberExtractor). It is still LLM-free: this script only runs the
extraction half.

Usage:
    python scripts/extract_pdfplumber.py
    python scripts/extract_pdfplumber.py --start 3 --end 5
    python scripts/extract_pdfplumber.py --out-dir some/where
"""
import argparse
import json
from pathlib import Path

from dmpbridge.extractors import get_extractor


def extract_range(extractor, pdf_dir: Path, out_dir: Path, start: int, end: int) -> list[dict]:
    """Extract samples *start*..*end* and write one JSON per sample."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(start, end + 1):
        pdf = pdf_dir / f"sample{i}.pdf"
        if not pdf.exists():
            print(f"  sample{i:<3} SKIPPED — {pdf} not found")
            continue

        blocks = extractor.extract(pdf)
        out_path = out_dir / f"sample{i}.json"
        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        chars = sum(len(b.get("text", "")) for b in blocks)
        bold  = sum(b.get("text", "").count("**") // 2 for b in blocks)
        ital  = sum(b.get("text", "").count("_") // 2 for b in blocks)
        rows.append({"sample": i, "chars": chars, "bold": bold, "italic": ital})
        print(f"  sample{i:<3} {chars:>6} chars  {bold:>3} bold run(s)  "
              f"{ital:>3} italic run(s)  -> {out_path}")
    return rows


def summarise(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    total = sum(r["chars"] for r in rows)
    print(f"\n  {label}: {total} chars across {len(rows)} document(s), "
          f"mean {total / len(rows):.0f} chars per document")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=10)
    ap.add_argument("--pdf-dir", type=Path, default=Path("data/input/pdfs"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/output/extracted"))
    args = ap.parse_args()

    print("=== pdfplumber ===")
    extractor = get_extractor("pdfplumber")
    rows = extract_range(extractor, args.pdf_dir, args.out_dir / "pdfplumber",
                         args.start, args.end)
    summarise("pdfplumber", rows)


if __name__ == "__main__":
    main()
