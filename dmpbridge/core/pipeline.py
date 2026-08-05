"""Main pipeline: PDF → labeled blocks → structured JSON.

The pipeline has two execution modes:

**Strategy mode** (recommended)
    Pass a :class:`~dmpbridge.strategies.Strategy` instance.  The strategy owns
    both preprocessing and model-call logic, so the pipeline stays thin::

        from dmpbridge.strategies import get_strategy
        strategy = get_strategy("wholedoc", model="llama3.3:70b")
        blocks   = process_pdf("document.pdf", strategy=strategy)

**Legacy mode** (backward compatible)
    Pass ``provider`` / ``model`` / ``host`` kwargs directly.  The pipeline
    creates a :class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy` internally::

        blocks = process_pdf("document.pdf", model="llama3.3:70b")

Both modes produce the same output: a flat list of labeled block dicts.
"""
import json
from pathlib import Path
from typing import Optional, Union

from ..preprocess import save_page_images
from ..utils import ExtractionError, get_logger
from . import config
from .converter import to_structured

logger = get_logger(__name__)

DEFAULT_PROVIDER = config.PROVIDER
DEFAULT_MODEL    = config.MODEL
DEFAULT_HOST     = config.HOST
DEFAULT_RAW_DIR  = "data/output/extracted"


def _write_outputs(
    blocks: list[dict],
    out_path: Optional[Path],
    struct_path: Optional[Path] = None,
    apply_rules: bool = False,
) -> None:
    """Write flat labeled JSON and/or structured JSON to disk."""
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if struct_path is not None:
        structured = to_structured(blocks)
        if apply_rules:
            from ..evaluation.annotation_rules import apply_new_annotation_rules
            structured = apply_new_annotation_rules(structured)
        struct_path.parent.mkdir(parents=True, exist_ok=True)
        struct_path.write_text(
            json.dumps(structured, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def run_and_save(
    strategy,
    pdf_path: Path,
    out_path: Path,
    struct_path: Optional[Path] = None,
    apply_rules: bool = False,
    final_path: Optional[Path] = None,
) -> Optional[list[dict]]:
    """Run *strategy* on one PDF and write each pipeline stage to its own path.

    Shared by the batch runners (``dmpbridge-wholedoc`` and
    ``dmpbridge-experiment``) so the save-and-convert logic lives in exactly
    one place instead of two independently-maintained copies.

    Parameters
    ----------
    out_path:
        Stage 2 — LLM-labeled blocks.
    struct_path:
        Stage 3 — the DMP Tool schema, without rule conversion.
    apply_rules:
        Legacy: apply the rules to *struct_path* itself.  Prefer *final_path*,
        which keeps stage 3 and stage 4 separate and inspectable.
    final_path:
        Stage 4 — the rule-converted result, written alongside an unconverted
        stage 3.  Takes precedence over *apply_rules*.

    Returns the labeled blocks, or ``None`` if *pdf_path* does not exist —
    callers are responsible for their own "already processed" skip check and
    logging, since they differ (progress/ETA logging vs. output-path lists).
    """
    if not pdf_path.exists():
        return None
    blocks = strategy.run(pdf_path)

    # With a separate stage 4, stage 3 must stay unconverted so the two can be
    # compared; applying the rules in place would collapse them into one.
    _write_outputs(blocks, out_path, struct_path,
                   apply_rules=apply_rules and final_path is None)

    if final_path is not None:
        from ..evaluation.annotation_rules import apply_new_annotation_rules
        final = apply_new_annotation_rules(to_structured(blocks))
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return blocks


def process_pdf(
    pdf_path: Union[str, Path],
    *,
    strategy=None,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    extractor: str = "pdfplumber",
    apply_rules: bool = False,
    output: Optional[Union[str, Path]] = None,
    structured_output: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = DEFAULT_RAW_DIR,
    images_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """Run the full pipeline for one PDF.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    strategy:
        A pre-built :class:`~dmpbridge.strategies.Strategy` instance.
        When provided, all other kwargs (model, extractor, etc.) are ignored.
    provider, model, host:
        LLM backend settings — ignored when *strategy* is given.
    extractor:
        PDF extraction backend: ``"pdfplumber"`` (default), ``"docling"``,
        or ``"lighton"``.  Ignored when *strategy* is given.
    apply_rules:
        When ``True``, apply the new-annotation rules (backfill empty
        question texts from section titles) to the structured JSON before
        saving.  Default ``False``.
    output:
        Path to write the flat labeled JSON.
    structured_output:
        Path to write the nested DMP Tool JSON.
    raw_dir:
        Directory to save extracted blocks before labeling.
        Set to ``None`` to skip.
    images_dir:
        Directory for per-page PNG images with bounding-box overlays.
        Set to ``None`` to skip.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError("PDF not found: %s" % pdf_path)

    # ── Resolve strategy ──────────────────────────────────────────────────────
    if strategy is None:
        from ..strategies.wholedoc import WholeDocStrategy
        strategy = WholeDocStrategy(provider=provider, model=model, host=host, extractor=extractor)

    # ── Extract + classify (delegated to strategy) ────────────────────────────
    logger.info("Running strategy %s on %s …", type(strategy).__name__, pdf_path.name)
    blocks = strategy.run(pdf_path)
    if not blocks:
        logger.warning("No text blocks found in %s", pdf_path.name)
        return []
    pages = len({b["page"] for b in blocks})
    logger.info("  → %d blocks across %d page(s)", len(blocks), pages)

    # ── Optional: save raw extracted blocks ───────────────────────────────────
    if raw_dir is not None:
        raw_path = Path(raw_dir) / f"{pdf_path.stem}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Extracted blocks saved → %s", raw_path)

    # ── Optional: save page images ────────────────────────────────────────────
    if images_dir is not None:
        logger.info("Saving page images → %s …", images_dir)
        try:
            saved = save_page_images(pdf_path, blocks, output_dir=images_dir)
            logger.info("  → %d image(s) saved", len(saved))
        except ExtractionError as exc:
            logger.warning("Image export skipped: %s", exc)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_path    = Path(output)            if output            else None
    struct_path = Path(structured_output) if structured_output else None
    _write_outputs(blocks, out_path, struct_path, apply_rules=apply_rules)
    if out_path:
        logger.info("Labeled JSON saved → %s", out_path)
    if struct_path:
        logger.info("Structured JSON saved → %s", struct_path)

    return blocks
