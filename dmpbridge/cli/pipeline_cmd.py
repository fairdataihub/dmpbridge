"""All six pipeline stages in one command.

Stages 1-4 are the narrative pipeline, run by ``dmpbridge-wholedoc`` itself,
unchanged; stages 5-6 (RDA maDMP JSON, then the merge) are added after it.

Usage:
    dmpbridge-pipeline --narrative-model gemma4:e4b --rda-model llama3.1:8b --start 3 --end 3

Writes, for each sample:
    1_extracted/<extractor>/                              plan text
    2_labeled|3_structured|4_final/<narrative-model>_<extractor>_whole_doc/
    5_rda/sampleN.json                                    RDA JSON
    6_merged/sampleN.json                                 {"dmp": ..., "narrative": ...}

(5_rda and 6_merged are not keyed by model: a sampleN.run.json beside each file
names the models that made it.)

Then, for each sample with a hand-made reference in data/input/rda_ground_truth/, the
RDA JSON is scored as in ``dmpbridge-evaluate-rda`` (the field matrix, and precision,
recall and F1); ``--no-evaluate`` skips it.

Every stage is cached by file existence, as in ``dmpbridge-wholedoc``: a sample
already done is skipped. Both parts read the plan's text the same way
(``--extractor``, with the ``--fallback`` extractor for an unreadable PDF).
"""
import argparse
import sys
from pathlib import Path

from ..core import config, paths
from ..core.pipeline import normalize_fallbacks
from ..rda.stages import run_rda_stages
from ..strategies.wholedoc import WholeDocStrategy
from ..utils import get_logger
from . import wholedoc_cmd

logger = get_logger(__name__)


def main() -> None:
    """CLI entry point: dmpbridge-pipeline."""
    ap = argparse.ArgumentParser(
        description="Run the narrative pipeline (stages 1-4) and the RDA extraction "
                    "(stages 5-6) on DMP PDF samples."
    )
    ap.add_argument("--narrative-model", required=True, help="model for stages 2-4")
    ap.add_argument("--rda-model", required=True, help="model for stage 5 (v10 RDA prompt)")
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
    ap.add_argument("--no-evaluate", action="store_true",
                    help="skip scoring the RDA JSON against the hand-made references")
    args = ap.parse_args()

    # Stages 1-4: the narrative pipeline exactly as `dmpbridge-wholedoc` runs it.
    sys.argv = ["dmpbridge-wholedoc", "--model", args.narrative_model, "--host", args.host,
                "--extractor", args.extractor, "--fallback", args.fallback,
                "--pdf-dir", str(args.pdf_dir), "--start", str(args.start), "--end", str(args.end)]
    wholedoc_cmd.main()

    # Stages 5-6. Only the strategy's text extraction is used, never its model.
    logger.info("=" * 55)
    logger.info("RDA model : %s  ->  5_rda/  ->  6_merged/", args.rda_model)
    logger.info("=" * 55)
    strategy = WholeDocStrategy(
        model=args.rda_model, host=args.host, extractor=args.extractor,
        cache_dir=paths.EXTRACTED_DIR / args.extractor,
        fallback_extractors=normalize_fallbacks(args.fallback, args.extractor),
    )
    for i in range(args.start, args.end + 1):
        run_rda_stages(strategy, args.pdf_dir / f"sample{i}.pdf", i, args.rda_model,
                       args.narrative_model, args.extractor, host=args.host)

    # Score the RDA JSON for the samples that have a hand-made reference.
    if not args.no_evaluate:
        from .rda_evaluate_cmd import evaluate_rda
        evaluate_rda(list(range(args.start, args.end + 1)))


if __name__ == "__main__":
    main()
