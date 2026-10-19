"""RDA maDMP extraction CLI - pipeline stages 5 and 6 on their own.

For a narrative run that already exists. To run all six stages in one command use
``dmpbridge-pipeline --narrative-model <model> --rda-model <model>`` instead.

Stage 5 reads each plan's text exactly as the narrative pipeline does (stage-1
cache, pdfplumber, LightOnOCR fallback for an unreadable PDF), sends it to
``--model`` with the v10 RDA prompt and the maDMP 1.2 schema, and writes the RDA
JSON. Stage 6 merges it with the narrative pipeline's stage-4 JSON for
``--narrative-model``.

Usage:
    dmpbridge-rda --model llama3.1:8b --narrative-model gemma4:e4b --start 3 --end 3

Writes: see dmpbridge.rda.stages.
"""
import argparse
from pathlib import Path

from ..core import config, paths
from ..core.pipeline import normalize_fallbacks
from ..rda import PROMPT_VERSION
from ..rda.stages import run_rda_stages
from ..strategies.wholedoc import WholeDocStrategy
from ..utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """CLI entry point: dmpbridge-rda."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Extract RDA maDMP JSON from DMP PDFs and merge it with the narrative JSON."
    )
    ap.add_argument("--model", required=True, help="Ollama model for the RDA extraction")
    ap.add_argument("--narrative-model", default=None,
                    help="model whose stage-4 narrative JSON to merge with (default: --model)")
    ap.add_argument("--host", default=config.HOST, help="Ollama host URL (default: %(default)s)")
    ap.add_argument("--extractor", default="pdfplumber",
                    choices=["pdfplumber", "lightonocr", "docling"],
                    help="PDF extraction backend (default: %(default)s)")
    ap.add_argument("--fallback", choices=["docling", "lightonocr", "auto"], default="auto",
                    help="extractor for a document whose text fails the garbled-text check "
                         "(default: %(default)s = lightonocr)")
    ap.add_argument("--pdf-dir", default="data/input/pdfs", type=Path)
    ap.add_argument("--start", default=1, type=int, help="First sample index (inclusive)")
    ap.add_argument("--end", default=10, type=int, help="Last sample index (inclusive)")
    args = ap.parse_args()
    narrative_model = args.narrative_model or args.model

    logger.info("=" * 55)
    logger.info("RDA model       : %s  (prompt %s)", args.model, PROMPT_VERSION)
    logger.info("Narrative model : %s", narrative_model)
    logger.info("Extractor       : %s  (fallback %s)", args.extractor, args.fallback)
    logger.info("Samples         : %d-%d", args.start, args.end)
    logger.info("Output          : 5_rda/  +  4_final/%s  ->  6_merged/",
                paths.make_tag(narrative_model, args.extractor))
    logger.info("=" * 55)

    # Only its text extraction is used; the model it is built with is never called.
    strategy = WholeDocStrategy(
        model=args.model, host=args.host, extractor=args.extractor,
        cache_dir=paths.EXTRACTED_DIR / args.extractor,
        fallback_extractors=normalize_fallbacks(args.fallback, args.extractor),
    )

    for i in range(args.start, args.end + 1):
        run_rda_stages(strategy, args.pdf_dir / f"sample{i}.pdf", i, args.model,
                       narrative_model, args.extractor, host=args.host)


if __name__ == "__main__":
    main()
