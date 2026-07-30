"""Command-line interface for dmpbridge."""

# What this file does — step by step:
#   Step 1 — parse command-line arguments (PDF path, model, output paths, flags)
#   Step 2 — validate that the input PDF file actually exists
#   Step 3 — resolve output paths for the flat JSON and structured JSON
#   Step 4 — call process_pdf() which runs the full pipeline (extract → classify → convert)
#   Step 5 — print a summary showing block counts per label and the paths of saved files

import argparse
import sys
from collections import Counter
from pathlib import Path

from ..core import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_RAW_DIR, process_pdf
from ..utils import DmpBridgeError, get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Entry point for the dmpbridge CLI.

    1. Parse command-line arguments.
    2. Validate the input PDF exists.
    3. Resolve output paths.
    4. Run the full pipeline.
    5. Print a summary of labeled block counts.
    """
    parser = argparse.ArgumentParser(
        prog="dmpbridge",
        description="Extract and label DMP PDF text blocks (batch strategy, all providers).",
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Labeled JSON output path. Defaults to <pdf_name>_labeled.json.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["ollama"],
        help=f"LLM provider to use (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for the selected provider (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=(
            f"Ollama server URL — only used with --provider ollama "
            f"(default: {DEFAULT_HOST})."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        metavar="DIR",
        help=(
            f"Folder for raw pdfplumber JSON saved before LLM labeling "
            f"(default: {DEFAULT_RAW_DIR})."
        ),
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
            "Produced by default as <output_stem>_structured.json. "
            "Pass a path to override the location."
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

    setup_logging(verbose=args.verbose)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    output = (
        Path(args.output)
        if args.output
        else pdf_path.with_name(pdf_path.stem + "_labeled.json")
    )
    raw_dir = None if args.no_raw else args.raw_dir

    if args.no_structured:
        structured_output = None
    else:
        structured_output = (
            Path(args.structured)
            if args.structured
            else output.with_name(output.stem + "_structured.json")
        )

    from ..strategies.batch import BatchStrategy
    strategy = BatchStrategy(provider=args.provider, model=args.model, host=args.host)

    try:
        blocks = process_pdf(
            pdf_path,
            strategy=strategy,
            output=output,
            structured_output=structured_output,
            raw_dir=raw_dir,
            images_dir=args.save_images,
        )
    except DmpBridgeError as exc:
        logger.error("%s", exc)
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
