"""Whole-document inference CLI.

Runs the WholeDocStrategy over a set of sample PDFs and writes one labeled
JSON file per sample.

Usage:
    dmpbridge-wholedoc
    dmpbridge-wholedoc --model llama3.3:70b
    dmpbridge-wholedoc --model gemma4:e4b --extractor lighton
    dmpbridge-wholedoc --start 3 --end 6
"""
import argparse
import time
from pathlib import Path

from ..core import config, paths
from ..core.pipeline import run_and_save
from ..strategies.wholedoc import WholeDocStrategy
from ..utils import get_logger, setup_logging

logger = get_logger(__name__)


def _log_device_info(extractor: str) -> None:
    """Print info for all available GPUs before the run starts."""
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
        lines = [l for l in out.splitlines() if l.strip()]
        logger.info("GPUs : %d device(s) detected", len(lines))
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                idx, name, total, free = parts
                logger.info(
                    "  [%s] %s — %s MB total | %s MB free",
                    idx, name, total, free,
                )
        logger.info("       Ollama will auto-select GPU layers")
    except Exception:
        logger.info("GPU  : nvidia-smi not found — check CUDA drivers")


def main() -> None:
    """CLI entry point: dmpbridge-wholedoc."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Run whole-document LLM classification on DMP PDF samples."
    )
    ap.add_argument(
        "--provider", default=config.PROVIDER, choices=["ollama"],
        help="LLM provider (default: %(default)s)",
    )
    ap.add_argument("--model",   default=config.MODEL, help="Model name (default: %(default)s)")
    ap.add_argument("--host",    default=config.HOST,
                    help="Ollama host URL (default: %(default)s)")
    ap.add_argument(
        "--extractor", default="pdfplumber",
        choices=["pdfplumber", "lighton"],
        help="PDF extraction backend (default: %(default)s)",
    )
    ap.add_argument("--pdf-dir", default="data/input/pdfs", type=Path)
    ap.add_argument("--start",       default=1,  type=int, help="First sample index (inclusive)")
    ap.add_argument("--end",         default=10, type=int, help="Last sample index (inclusive)")
    ap.add_argument("--no-rules", action="store_true",
                    help="Skip stage 4 — do not write the rule-converted final JSON")
    ap.add_argument("--no-cache", action="store_true",
                    help="Re-extract even when stage 1 already holds this sample")
    args = ap.parse_args()

    n_samples = args.end - args.start + 1
    tag       = paths.make_tag(args.model, args.extractor)

    logger.info("=" * 55)
    logger.info("Model     : %s", args.model)
    logger.info("Extractor : %s", args.extractor)
    logger.info("Ollama    : %s", args.host)
    logger.info("Samples   : %d–%d  (%d total)", args.start, args.end, n_samples)
    logger.info("Tag       : %s", tag)
    logger.info("Output    : 1_extracted/%s  ->  2_labeled|3_structured|4_final/%s",
                args.extractor, tag)
    _log_device_info(args.extractor)
    logger.info("=" * 55)

    cache_dir = None if args.no_cache else paths.EXTRACTED_DIR / args.extractor
    strategy = WholeDocStrategy(
        provider=args.provider,
        model=args.model,
        host=args.host,
        extractor=args.extractor,
        cache_dir=cache_dir,
    )

    run_start   = time.perf_counter()
    done        = 0
    sample_times: list[float] = []

    for i in range(args.start, args.end + 1):
        label       = f"[sample{i}]"
        pdf_path    = args.pdf_dir / f"sample{i}.pdf"
        out_path    = paths.labeled_path(tag, i)
        struct_path = paths.structured_path(tag, i)
        final_path  = None if args.no_rules else paths.final_path(tag, i)

        if out_path.exists():
            logger.info("%s already exists — skipping", label)
            continue

        t0      = time.perf_counter()
        blocks  = run_and_save(strategy, pdf_path, out_path, struct_path,
                               final_path=final_path)
        elapsed = time.perf_counter() - t0

        if blocks is None:
            logger.warning("%s PDF not found: %s", label, pdf_path)
            continue

        sample_times.append(elapsed)
        done += 1

        remaining = n_samples - done
        if sample_times and remaining > 0:
            avg     = sum(sample_times) / len(sample_times)
            eta_s   = avg * remaining
            eta_str = f"{eta_s/60:.1f} min" if eta_s >= 60 else f"{eta_s:.0f} s"
            logger.info(
                "%s done in %.1f s  |  %d/%d  |  ETA ~%s",
                label, elapsed, done, n_samples, eta_str,
            )
        else:
            logger.info("%s done in %.1f s  |  %d/%d", label, elapsed, done, n_samples)
        stages = "1_extracted -> 2_labeled -> 3_structured" + \
                 ("" if final_path is None else " -> 4_final")
        logger.info("%s %s", label, stages)

    total = time.perf_counter() - run_start
    logger.info("=" * 55)
    if sample_times:
        avg = sum(sample_times) / len(sample_times)
        logger.info(
            "Done  %d samples  |  total %.1f s  |  avg %.1f s/sample",
            done, total, avg,
        )
    else:
        logger.info("Done — all samples already existed, nothing to run.")


if __name__ == "__main__":
    main()
