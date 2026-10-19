"""RDA maDMP extraction CLI - pipeline stages 5 and 6.

Stage 5 reads each plan's text exactly as the narrative pipeline does (stage-1
cache, pdfplumber, LightOnOCR fallback for an unreadable PDF), sends it to
``--model`` with the v10 RDA prompt and the maDMP 1.2 schema, and writes the RDA
JSON. Stage 6 merges it with the narrative pipeline's stage-4 JSON for
``--narrative-model``, which must already have been run (``dmpbridge-wholedoc``).

Usage:
    dmpbridge-rda --model llama3.1:8b --narrative-model gemma4:e4b --start 3 --end 3

Writes:
    5_rda/<model>_<extractor>_rda/sampleN.json              the RDA JSON
    5_rda/<model>_<extractor>_rda/sampleN.raw.txt           the answer, when it is not valid JSON
    6_merged/<narrative-model>+<model>_<extractor>/sampleN.json   {"narrative": ..., "dmp": ...}

Like the narrative stages, stage 5 is cached by file existence: delete
``sampleN.json`` to re-run a sample. Stage 6 is rewritten every run, since it is
only a merge.
"""
import argparse
import json
import time
from pathlib import Path

from ..core import config, paths
from ..core.pipeline import normalize_fallbacks
from ..rda import PROMPT_VERSION, RdaExtractionError, extract_rda, merge
from ..strategies.wholedoc import WholeDocStrategy
from ..utils import get_logger, setup_logging

logger = get_logger(__name__)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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

    rda_tag = paths.rda_tag(args.model, args.extractor)
    narrative_tag = paths.make_tag(narrative_model, args.extractor)
    merged_tag = paths.merged_tag(narrative_model, args.model, args.extractor)

    logger.info("=" * 55)
    logger.info("RDA model       : %s  (prompt %s)", args.model, PROMPT_VERSION)
    logger.info("Narrative model : %s", narrative_model)
    logger.info("Extractor       : %s  (fallback %s)", args.extractor, args.fallback)
    logger.info("Samples         : %d-%d", args.start, args.end)
    logger.info("Output          : 5_rda/%s  +  4_final/%s  ->  6_merged/%s",
                rda_tag, narrative_tag, merged_tag)
    logger.info("=" * 55)

    # Only its text extraction is used; the model it is built with is never called.
    strategy = WholeDocStrategy(
        model=args.model, host=args.host, extractor=args.extractor,
        cache_dir=paths.EXTRACTED_DIR / args.extractor,
        fallback_extractors=normalize_fallbacks(args.fallback, args.extractor),
    )

    for i in range(args.start, args.end + 1):
        label = f"[sample{i}]"
        pdf_path = args.pdf_dir / f"sample{i}.pdf"
        rda_out = paths.rda_path(rda_tag, i)

        if rda_out.exists():
            logger.info("%s RDA JSON already exists - skipping extraction", label)
            rda = json.loads(rda_out.read_text(encoding="utf-8"))
        elif not pdf_path.exists():
            logger.warning("%s PDF not found: %s", label, pdf_path)
            continue
        else:
            text = strategy.extract_text(pdf_path)
            if not text:
                continue
            t0 = time.perf_counter()
            try:
                rda, info = extract_rda(text, args.model, host=args.host)
            except RdaExtractionError as e:
                raw_out = rda_out.with_name(f"sample{i}.raw.txt")
                raw_out.parent.mkdir(parents=True, exist_ok=True)
                raw_out.write_text(e.raw, encoding="utf-8")
                logger.error("%s %s - answer saved to %s", label, e, raw_out)
                continue
            _write_json(rda_out, rda)
            logger.info("%s RDA JSON in %.1f s (%s output tokens) -> %s",
                        label, time.perf_counter() - t0, info.get("eval_count"), rda_out)

        narrative_in = paths.final_path(narrative_tag, i)
        if not narrative_in.exists():
            logger.warning("%s no narrative JSON at %s - run dmpbridge-wholedoc --model %s "
                           "first; not merged", label, narrative_in, narrative_model)
            continue
        narrative = json.loads(narrative_in.read_text(encoding="utf-8"))
        merged_out = paths.merged_path(merged_tag, i)
        _write_json(merged_out, merge(narrative, rda))
        logger.info("%s merged -> %s", label, merged_out)


if __name__ == "__main__":
    main()
