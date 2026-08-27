"""Write an extractor's raw native reading of each sample PDF as one JSON file.

Thin wrapper over ``<Extractor>(save_native=True)``: runs the pipeline's own
extractor with the native dump switched on, so each sample gets a
``sampleN.native.json`` next to its stage-1 text under
``data/output/1_extracted/<extractor>/``.

    docling      Docling's full conversion result — document, layout clusters
                 with confidences, parsed page cells (font, size, position per
                 line and word), hyperlinks with URLs, page images
    pdfplumber   everything pdfplumber reads before the marker rules — words
                 with font name, size and box; rectangles, lines, curves
                 (underline evidence); hyperlinks with URLs; body-font profile

Neither changes the stage-1 text; the native file is evidence for tracing a
marker back to what produced it.

    python scripts/native_dump.py --extractor pdfplumber          # samples 1-10
    python scripts/native_dump.py --extractor docling --no-images
    python scripts/native_dump.py --extractor pdfplumber --chars  # include every character
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dmpbridge.core import paths as P
from dmpbridge.extractors import get_extractor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="docling", choices=["docling", "pdfplumber"])
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--no-images", action="store_true", help="docling: leave the page images out")
    ap.add_argument("--chars", action="store_true", help="pdfplumber: include every character")
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.disable(logging.WARNING)
    kwargs = ({"native_images": not args.no_images} if args.extractor == "docling"
              else {"native_chars": args.chars})
    extractor = get_extractor(args.extractor, save_native=True, **kwargs)

    for n in range(args.start, args.end + 1):
        extractor.extract(Path("data/input/pdfs") / f"sample{n}.pdf")
        out = P.EXTRACTED_DIR / args.extractor / f"sample{n}.native.json"
        native = json.loads(out.read_text(encoding="utf-8"))
        pages = native["pages"]
        if args.extractor == "docling":
            units = sum(len((p.get("parsed_page") or {}).get("textline_cells", [])) for p in pages)
            links = sum(len((p.get("parsed_page") or {}).get("hyperlinks", [])) for p in pages)
            detail = f"{units:4d} text lines  {links:3d} hyperlinks"
        else:
            units = sum(len(p["words"]) for p in pages)
            rects = sum(len(p["rects"]) for p in pages)
            links = sum(len(p["hyperlinks"]) for p in pages)
            detail = f"{units:4d} words  {rects:3d} rects  {links:3d} hyperlinks"
        print(f"sample{n:<3} {len(pages)} pages  {detail}  "
              f"{out.stat().st_size / 1024:6.0f} KB  -> {out.name}")


if __name__ == "__main__":
    main()
