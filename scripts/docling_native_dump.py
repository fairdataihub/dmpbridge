"""Write Docling's full native conversion result for each sample as one JSON file.

Thin wrapper over ``DoclingExtractor(save_native=True)``: runs the pipeline's
own Docling extractor with the native dump switched on, so each sample ends up
with three views under ``data/output/1_extracted/docling/``:

    sampleN.json           the text the pipeline sends to the model
    sampleN.md             Docling's untranslated Markdown
    sampleN.native.json    Docling's full native result — parsed page cells
                           (font, size, position per line and word),
                           hyperlinks with URLs, layout clusters with
                           confidences, page images, confidence report

The native file is what ``ConversionResult.save()`` produces, merged from its
zip parts into one JSON (see ``native_result_dict`` in the extractor). Page
images are most of the size (~2–5 MB per document with them, a few hundred KB
without); pass --no-images to leave them out.

    python scripts/docling_native_dump.py                 # samples 1-10
    python scripts/docling_native_dump.py --start 2 --end 2
    python scripts/docling_native_dump.py --no-images
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dmpbridge.core import paths as P
from dmpbridge.extractors.docling_extractor import DoclingExtractor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--no-images", action="store_true", help="leave the page images out")
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.disable(logging.WARNING)
    extractor = DoclingExtractor(save_native=True, native_images=not args.no_images)

    for n in range(args.start, args.end + 1):
        extractor.extract(Path("data/input/pdfs") / f"sample{n}.pdf")
        out = P.EXTRACTED_DIR / "docling" / f"sample{n}.native.json"
        native = json.loads(out.read_text(encoding="utf-8"))
        pages = native["pages"]
        lines = sum(len((p.get("parsed_page") or {}).get("textline_cells", [])) for p in pages)
        links = sum(len((p.get("parsed_page") or {}).get("hyperlinks", [])) for p in pages)
        print(f"sample{n:<3} {len(pages)} pages  {lines:4d} text lines  {links:3d} hyperlinks  "
              f"{out.stat().st_size / 1024 / 1024:5.1f} MB  -> {out.name}")


if __name__ == "__main__":
    main()
