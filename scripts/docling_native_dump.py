"""Write Docling's full native conversion result for each sample as one JSON file.

The pipeline's Docling extractor keeps only the Markdown-derived text
(``sampleN.json``) and the raw Markdown (``sampleN.md``). Docling discards the
levels underneath — the parsed page cells with font names, sizes, positions
and hyperlinks, the layout model's clusters with confidences, the page images
— unless asked to keep them. This script asks, and writes everything Docling
serialises (``ConversionResult.save``, which produces a zip of JSON parts) as
a single ``sampleN.native.json`` next to the other two, so every sample has
three views: pipeline text, Markdown, native.

Structure of the file:

    version, status, timestamp, timings, errors
    confidence            per-page layout / parse scores
    document              the DoclingDocument (what export_to_dict() returns)
    pages[]               per page: size, predictions.layout.clusters,
                          assembled, parsed_page{textline_cells, word_cells,
                          hyperlinks, image, ...}

Page images are most of the size (~5 MB per document with them, a few hundred
KB without). They are included by default to match Docling's own save();
pass --no-images to leave them out.

    python scripts/docling_native_dump.py                 # samples 1-10
    python scripts/docling_native_dump.py --start 2 --end 2
    python scripts/docling_native_dump.py --no-images
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

from dmpbridge.core import paths as P

OUT_DIR = P.EXTRACTED_DIR / "docling"
PART_ORDER = ("version", "status", "timestamp", "timings", "errors",
              "confidence", "document", "pages")


def make_converter(images: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True                        # same as the pipeline's extractor
    opts.do_table_structure = True
    opts.generate_parsed_pages = True         # keep cells: fonts, sizes, words, links
    opts.generate_page_images = images        # keep the page renders
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def native_dict(result) -> dict:
    """Docling's own serialisation (a zip of JSON parts), merged into one dict."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "native.zip"
        result.save(filename=bundle)
        with zipfile.ZipFile(bundle) as z:
            parts = {name[:-5]: json.loads(z.read(name).decode("utf-8"))
                     for name in z.namelist() if name.endswith(".json")}
    ordered = {k: parts[k] for k in PART_ORDER if k in parts}
    ordered.update({k: v for k, v in parts.items() if k not in ordered})
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--no-images", action="store_true", help="leave the page images out")
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.disable(logging.WARNING)
    converter = make_converter(images=not args.no_images)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in range(args.start, args.end + 1):
        result = converter.convert(str(Path("data/input/pdfs") / f"sample{n}.pdf"))
        native = native_dict(result)
        out = OUT_DIR / f"sample{n}.native.json"
        out.write_text(json.dumps(native, indent=2, ensure_ascii=False), encoding="utf-8")
        pages = native["pages"]
        lines = sum(len((p.get("parsed_page") or {}).get("textline_cells", [])) for p in pages)
        links = sum(len((p.get("parsed_page") or {}).get("hyperlinks", [])) for p in pages)
        print(f"sample{n:<3} {len(pages)} pages  {lines:4d} text lines  {links:3d} hyperlinks  "
              f"{out.stat().st_size / 1024 / 1024:5.1f} MB  -> {out.name}")


if __name__ == "__main__":
    main()
