"""Command-line interface for dmpbridge."""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from .pipeline import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_RAW_DIR, process_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dmpbridge",
        description="Extract and label PDF text blocks using pdfplumber + LLaMA via Ollama.",
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Labeled JSON output path. Defaults to <pdf_name>_labeled.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama server URL (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        metavar="DIR",
        help=f"Folder for raw pdfplumber JSON saved before LLM labeling (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip saving the raw pdfplumber extraction JSON.",
    )
    parser.add_argument(
        "--save-images",
        default=None,
        metavar="DIR",
        help="Also save per-page PNG images with bounding boxes to this folder.",
    )
    parser.add_argument(
        "--structured",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Also write a hierarchical structured JSON (dmpchef schema). "
            "If PATH is omitted, saves as <output_stem>_structured.json alongside the flat output."
        ),
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Skip post-processing smoothing rules (useful for ablation testing).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress logs.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found — {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output  = Path(args.output) if args.output else pdf_path.with_name(pdf_path.stem + "_labeled.json")
    raw_dir = None if args.no_raw else args.raw_dir

    structured_output = None
    if args.structured is not None:
        structured_output = (
            Path(args.structured)
            if args.structured
            else output.with_name(output.stem + "_structured.json")
        )

    try:
        blocks = process_pdf(
            pdf_path,
            model=args.model,
            host=args.host,
            output=output,
            structured_output=structured_output,
            raw_dir=raw_dir,
            images_dir=args.save_images,
            smooth=not args.no_smooth,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    counts = Counter(b.get("label", "answer.text") for b in blocks)
    print(f"\nDone — {len(blocks)} blocks labeled:")
    for lbl in ("title", "section.title", "section.description", "question.text", "answer.text"):
        n = counts.get(lbl, 0)
        if n:
            print(f"  {lbl:<22} {n}")
    if raw_dir:
        print(f"\nRaw extraction : {raw_dir}/{pdf_path.stem}.json")
    print(f"Labeled output : {output}")
    if structured_output:
        print(f"Structured JSON: {structured_output}")


if __name__ == "__main__":
    main()
