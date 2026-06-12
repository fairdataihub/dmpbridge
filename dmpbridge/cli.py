"""Command-line interface for dmpbridge."""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from .pipeline import process_pdf, DEFAULT_MODEL, DEFAULT_HOST


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dmpbridge",
        description="Extract and label PDF text blocks using pdfplumber + LLaMA via Ollama.",
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file path. Defaults to <pdf_name>_labeled.json.",
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

    output = args.output or pdf_path.with_name(pdf_path.stem + "_labeled.json")

    try:
        blocks = process_pdf(
            pdf_path,
            model=args.model,
            host=args.host,
            output=output,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    counts = Counter(b.get("label", "content") for b in blocks)
    print(f"\nDone — {len(blocks)} blocks labeled:")
    for lbl in ("document_title", "section", "subsection", "content"):
        n = counts.get(lbl, 0)
        if n:
            print(f"  {lbl:<20} {n}")
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
