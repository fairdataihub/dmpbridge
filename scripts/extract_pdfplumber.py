"""Extract text blocks from the sample PDFs with pdfplumber — no LLM, no labeling.

Writes one JSON file per sample containing the blocks exactly as the extractor
produces them, which is what the classifier would receive as input.

Usage:
    python scripts/extract_pdfplumber.py                  # merged (pipeline default)
    python scripts/extract_pdfplumber.py --raw            # line-level, no merging
    python scripts/extract_pdfplumber.py --both           # write both, side by side
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

        pages = len({b["page"] for b in blocks}) if blocks else 0
        short = sum(1 for b in blocks if len(b["text"].strip()) <= 25)
        bold  = sum(1 for b in blocks if b.get("is_bold"))
        rows.append({"sample": i, "blocks": len(blocks), "pages": pages,
                     "short": short, "bold": bold})
        print(f"  sample{i:<3} {len(blocks):>4} blocks  {pages} page(s)  "
              f"{short:>3} short  {bold:>3} bold  -> {out_path}")
    return rows


def summarise(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    total = sum(r["blocks"] for r in rows)
    print(f"\n  {label}: {total} blocks across {len(rows)} document(s), "
          f"mean {total / len(rows):.1f} per document, "
          f"{sum(r['short'] for r in rows)} short fragments (<= 25 chars)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true",
                    help="Disable line merging — one block per PDF line")
    ap.add_argument("--both", action="store_true",
                    help="Write merged and raw side by side for comparison")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=10)
    ap.add_argument("--pdf-dir", type=Path, default=Path("data/input/pdfs"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/output/extracted"),
                    help="Parent directory; a subfolder is created per variant")
    args = ap.parse_args()

    variants = [("merged", True), ("raw", False)] if args.both \
        else [("raw", False)] if args.raw else [("merged", True)]

    results = {}
    for name, merge in variants:
        print(f"\n=== pdfplumber ({name}, merge_lines={merge}) ===")
        extractor = get_extractor("pdfplumber", merge_lines=merge)
        out_dir = args.out_dir / f"pdfplumber_{name}"
        results[name] = extract_range(extractor, args.pdf_dir, out_dir,
                                      args.start, args.end)
        summarise(name, results[name])

    if len(results) == 2:
        m, r = results["merged"], results["raw"]
        tm, tr = sum(x["blocks"] for x in m), sum(x["blocks"] for x in r)
        print(f"\n=== merged vs raw ===")
        print(f"  {'sample':<9}{'raw':>7}{'merged':>9}{'reduction':>12}")
        for a, b in zip(r, m):
            print(f"  {a['sample']:<9}{a['blocks']:>7}{b['blocks']:>9}"
                  f"{1 - b['blocks'] / a['blocks']:>11.0%}")
        print(f"  {'TOTAL':<9}{tr:>7}{tm:>9}{1 - tm / tr:>11.0%}")


if __name__ == "__main__":
    main()
