"""Command-line interface for dmpbridge."""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from .pipeline import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_RAW_DIR, process_pdf


def main() -> None:
    """Entry point for the dmpbridge command. 1. Parse command-line arguments. 2. Validate the input PDF exists. 3. Resolve output paths. 4. Run the full pipeline. 5. Print a summary of labeled block counts."""

    # Define all the arguments the user can pass from the terminal.
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
        default="",
        metavar="PATH",
        help=(
            "Path for the hierarchical structured JSON (DMP Tool narrative schema). "
            "Produced by default as <output_stem>_structured.json. Pass a path to override location."
        ),
    )
    parser.add_argument(
        "--no-structured",
        action="store_true",
        help="Skip writing the structured JSON.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress logs.",
    )

    args = parser.parse_args()

    # Set log level — verbose mode shows every batch being classified, normal mode just shows steps.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Validate that the PDF actually exists before doing any work.
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found — {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve output paths — default flat JSON goes next to the PDF with _labeled suffix.
    output  = Path(args.output) if args.output else pdf_path.with_name(pdf_path.stem + "_labeled.json")
    raw_dir = None if args.no_raw else args.raw_dir

    # Resolve structured JSON path — skip if --no-structured, otherwise default next to flat JSON.
    if args.no_structured:
        structured_output = None
    else:
        structured_output = (
            Path(args.structured)
            if args.structured
            else output.with_name(output.stem + "_structured.json")
        )

    # Run the full pipeline — extract, classify, and save.
    try:
        blocks = process_pdf(
            pdf_path,
            model=args.model,
            host=args.host,
            output=output,
            structured_output=structured_output,
            raw_dir=raw_dir,
            images_dir=args.save_images,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print a summary showing how many blocks got each label.
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
