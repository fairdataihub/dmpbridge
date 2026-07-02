"""Main pipeline: PDF → labeled blocks → structured JSON.

The pipeline has two execution modes:

**Strategy mode** (recommended)
    Pass a :class:`~dmpbridge.strategies.Strategy` instance.  The strategy owns
    both preprocessing and model-call logic, so the pipeline stays thin::

        from dmpbridge.strategies import get_strategy
        strategy = get_strategy("batch", provider="anthropic", model="claude-opus-4-8")
        blocks   = process_pdf("document.pdf", strategy=strategy)

**Legacy mode** (backward compatible)
    Pass ``provider`` / ``model`` / ``host`` kwargs directly.  The pipeline
    creates a :class:`~dmpbridge.strategies.batch.BatchStrategy` internally —
    exactly the previous behaviour::

        blocks = process_pdf("document.pdf", provider="anthropic", model="claude-opus-4-8")

Both modes produce the same output: a flat list of labeled block dicts.
"""
import json
from pathlib import Path
from typing import Optional, Union

from . import config
from .converter import to_structured
from .exceptions import ExtractionError
from .extractor import extract_blocks, save_page_images
from .logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_PROVIDER = config.PROVIDER
DEFAULT_MODEL    = config.MODEL
DEFAULT_HOST     = config.HOST
DEFAULT_RAW_DIR  = "data/pdfplumber"


def process_pdf(
    pdf_path: Union[str, Path],
    *,
    strategy=None,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
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
        A :class:`~dmpbridge.strategies.Strategy` instance.  When provided,
        ``provider`` / ``model`` / ``host`` are ignored for classification.
        The raw extraction step (pdfplumber save + images) is skipped when the
        strategy handles its own preprocessing (e.g. ``PdfDirectStrategy``).
    provider, model, host:
        Used in legacy mode (no strategy) to build a ``BatchStrategy``.
    output:
        Path to write the flat labeled JSON.
    structured_output:
        Path to write the nested DMP Tool JSON.
    raw_dir:
        Directory for the raw pdfplumber JSON (before labeling).
        Set to ``None`` to skip.  Ignored when *strategy* handles preprocessing.
    images_dir:
        Directory for per-page PNG images with bounding-box overlays.
        Set to ``None`` to skip.  Ignored when *strategy* handles preprocessing.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError("PDF not found: %s" % pdf_path)

    # ── Resolve strategy ──────────────────────────────────────────────────────
    if strategy is None:
        from .strategies.batch import BatchStrategy
        strategy = BatchStrategy(provider=provider, model=model, host=host)

    # ── Check whether this strategy does its own preprocessing ────────────────
    from .strategies.pdf_direct import PdfDirectStrategy
    _strategy_owns_preprocessing = isinstance(strategy, PdfDirectStrategy)

    # ── Preprocessing (pdfplumber path only) ──────────────────────────────────
    if not _strategy_owns_preprocessing:
        logger.info("Extracting text from %s …", pdf_path.name)
        _raw_blocks = extract_blocks(pdf_path)
        if not _raw_blocks:
            logger.warning("No text blocks found in the PDF.")
            return []
        pages = len({b["page"] for b in _raw_blocks})
        logger.info("  → %d blocks across %d page(s)", len(_raw_blocks), pages)

        if raw_dir is not None:
            raw_path = Path(raw_dir) / f"{pdf_path.stem}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(_raw_blocks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Raw extraction saved → %s", raw_path)

        if images_dir is not None:
            logger.info("Saving page images → %s …", images_dir)
            try:
                saved = save_page_images(pdf_path, _raw_blocks, output_dir=images_dir)
                logger.info("  → %d image(s) saved", len(saved))
            except ExtractionError as exc:
                logger.warning("Image export skipped: %s", exc)

    # ── Classification (delegated to strategy) ────────────────────────────────
    logger.info("Running strategy %s …", type(strategy).__name__)
    blocks = strategy.run(pdf_path)

    # ── Save flat labeled JSON ────────────────────────────────────────────────
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Labeled JSON saved → %s", out_path)

    # ── Save structured JSON ──────────────────────────────────────────────────
    if structured_output:
        struct_path = Path(structured_output)
        struct_path.parent.mkdir(parents=True, exist_ok=True)
        struct_path.write_text(
            json.dumps(to_structured(blocks), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Structured JSON saved → %s", struct_path)

    return blocks
