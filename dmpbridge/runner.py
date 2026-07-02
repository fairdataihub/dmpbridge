"""Whole-document inference CLI.

Runs the WholeDocStrategy (pdfplumber extraction + single LLM call) over a
batch of sample PDFs and writes one labeled JSON file per sample.

Usage:
    dmpbridge-wholedoc
    dmpbridge-wholedoc --provider ollama   --model llama3.3:70b
    dmpbridge-wholedoc --provider anthropic --model claude-opus-4-8
    dmpbridge-wholedoc --start 3 --end 6
"""
import argparse
import json
from pathlib import Path

from . import config
from .logging_setup import get_logger, setup_logging
from .strategies.wholedoc import WholeDocStrategy

logger = get_logger(__name__)


def main() -> None:
    """CLI entry point: dmpbridge-wholedoc."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Run whole-document LLM classification on DMP PDF samples."
    )
    ap.add_argument(
        "--provider", default=config.PROVIDER, choices=["anthropic", "ollama"],
        help="LLM provider (default: %(default)s)",
    )
    ap.add_argument("--model",   default=config.MODEL, help="Model name (default: %(default)s)")
    ap.add_argument("--host",    default=config.HOST,  help="Ollama host URL (default: %(default)s)")
    ap.add_argument("--pdf-dir", default="data/pdfsamples", type=Path)
    ap.add_argument("--out-dir", default="data/llmlabeled",  type=Path)
    ap.add_argument("--start",   default=1,  type=int, help="First sample index (inclusive)")
    ap.add_argument("--end",     default=10, type=int, help="Last sample index (inclusive)")
    args = ap.parse_args()

    strategy = WholeDocStrategy(
        provider=args.provider,
        model=args.model,
        host=args.host,
    )

    tag     = f"{args.model.replace(':', '-')}_whole_doc"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.start, args.end + 1):
        label    = f"[sample{i}]"
        pdf_path = args.pdf_dir / f"sample{i}.pdf"
        out_path = out_dir / f"sample{i}_{tag}.json"

        if out_path.exists():
            logger.info("%s already exists — skipping", label)
            continue

        if not pdf_path.exists():
            logger.warning("%s PDF not found: %s", label, pdf_path)
            continue

        blocks = strategy.run(pdf_path)

        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("%s saved → %s", label, out_path.name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
