"""Two-step extraction with quality-triggered fallback.

The architecture in one sentence: extract with the primary (pdfplumber for
digital PDFs), *validate* what came out, and only when validation fails —
broken text layer, parsing error, empty character set, undecodable font
encoding — route the document to a fallback extractor (LightOnOCR reading
page images, or Docling with forced full-page OCR).

    primary.extract() ──ok──► validate ──ok──► ExtractionResult(primary text)
          │error                  │reason
          └──────────────────────►┴──► for each fallback, in order:
                                         extract (fail-soft) ──► validate
                                         ──ok──► ExtractionResult(fallback text)
                                       all failed ──► ExtractionResult(primary
                                                      text, ok=False, reason)

Everything is observable: the result records which extractor's text was used
and every attempt with its outcome, so a caller (CLI log line today, a web
response tomorrow) can surface "this document was processed via OCR".

This module owns the flow; :mod:`~dmpbridge.preprocess.text_quality` owns the
validation heuristics; :class:`~dmpbridge.strategies.wholedoc.WholeDocStrategy`
delegates to :func:`extract_with_fallback` when fallbacks are configured.
Used standalone::

    from dmpbridge.extractors import get_extractor
    from dmpbridge.extractors.fallback import extract_with_fallback

    result = extract_with_fallback(Path("doc.pdf"),
                                   primary=get_extractor("pdfplumber"),
                                   fallback_names=["lightonocr", "docling"])
    result.text          # what to feed the model
    result.extractor     # who actually read the document
    result.ok            # False only if every extractor produced garbage
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.psparser import PSException

from ..preprocess.text_quality import looks_garbled
from .base import BaseExtractor

logger = logging.getLogger(__name__)

# Exceptions that mean "this PDF could not be parsed as a digital PDF" — a
# fallback trigger, not a crash. Anything else raised by the *primary* is a
# real bug and propagates.
PDF_PARSE_ERRORS: tuple[type[Exception], ...] = (
    PDFSyntaxError,        # damaged xref / malformed structure
    PSException,           # postscript-level parse failure
    PDFPasswordIncorrect,  # encrypted document
    ValueError,            # pdfplumber raises this on some malformed inputs
)


@dataclass(frozen=True)
class FallbackAttempt:
    """One extractor's try at a document: who, whether it was usable, and why not."""
    extractor: str
    ok: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class ExtractionResult:
    """The outcome of the two-step flow.

    ``ok=False`` means every extractor produced unusable text; ``text`` then
    holds the primary's output (possibly empty) and ``reason`` says what is
    wrong with it, so the caller can decide whether to proceed loudly or stop.
    """
    text: str
    extractor: str
    ok: bool
    reason: Optional[str] = None
    attempts: list[FallbackAttempt] = field(default_factory=list)


def _extractor_name(extractor: BaseExtractor) -> str:
    return getattr(extractor, "name", type(extractor).__name__)


def _page_count(pdf_path: Path) -> Optional[int]:
    """Page count for the per-page validation floor; None when unreadable."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return None


def _make_fallback(name: str) -> BaseExtractor:
    """Build a fallback extractor. Docling is always constructed with forced
    full-page OCR: a garbage text layer is precisely the case its automatic
    OCR does not catch (it only fires on pages with *no* text layer)."""
    from . import get_extractor
    kwargs = {"force_ocr": True} if name == "docling" else {}
    return get_extractor(name, **kwargs)


def _default_cache_write(name: str, pdf_path: Path, text: str) -> None:
    """Persist a successful fallback's text into that extractor's own stage-1
    directory (``1_extracted/<name>/<stem>.json``), same shape as any other
    stage-1 file.

    Two reasons this exists: the rescued text is inspectable on disk like
    every other extraction, and a rescued document becomes reproducible —
    later runs reuse this file via ``cache_lookup`` instead of re-rolling the
    OCR, whose output is only bit-stable within one machine state (observed
    2026-09-08: two rescues of the same document hours apart structured it
    differently, and the unstored morning text made the difference
    unexplainable). Best-effort: never fails the extraction.
    """
    try:
        import json
        from ..core.paths import EXTRACTED_DIR
        out = EXTRACTED_DIR / name / f"{pdf_path.stem}.json"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([{"text": text}], indent=2, ensure_ascii=False),
                           encoding="utf-8")
            logger.warning("[fallback] %s: %s text cached -> %s", pdf_path.name, name, out)
    except OSError:
        pass


def extract_with_fallback(
    pdf_path: Path,
    primary: BaseExtractor,
    fallback_names: Sequence[str],
    *,
    primary_text: Optional[str] = None,
    cache_lookup: Optional[Callable[[str], Optional[str]]] = None,
    make_extractor: Callable[[str], BaseExtractor] = _make_fallback,
    cache_write: Callable[[str, Path, str], None] = _default_cache_write,
) -> ExtractionResult:
    """Run the conditional two-step flow for one document.

    Parameters
    ----------
    primary:
        The extractor to trust first (normally pdfplumber).
    fallback_names:
        Extractor names to try, in order, when validation fails. Each attempt
        fails soft — an unavailable fallback (no CUDA GPU, missing extras) is
        recorded and skipped, not raised.
    primary_text:
        The primary's text when the caller already has it (e.g. from the
        stage-1 cache), so it is validated rather than re-extracted.
    cache_lookup:
        Optional ``name -> text`` for reusing a fallback extractor's own
        cached text instead of re-extracting; a cached text is still
        validated before use.
    make_extractor:
        Factory for fallback instances — injectable for tests.
    cache_write:
        Called with ``(name, pdf_path, text)`` when a fallback's text is
        accepted; by default persists it into that extractor's stage-1
        directory so the rescue is inspectable and reproducible.
    """
    attempts: list[FallbackAttempt] = []
    n_pages = _page_count(pdf_path)
    primary_name = _extractor_name(primary)

    # ── Step 1: the primary, with PDF-specific error handling ────────────────
    text = primary_text
    reason: Optional[str] = None
    if text is None:
        try:
            blocks = primary.extract(pdf_path)
            text = blocks[0]["text"] if blocks else ""
        except PDF_PARSE_ERRORS as exc:
            text = ""
            reason = f"{type(exc).__name__}: {exc}"
    if reason is None:
        reason = looks_garbled(text or "", n_pages)
    attempts.append(FallbackAttempt(primary_name, reason is None, reason))
    if reason is None:
        return ExtractionResult(text or "", primary_name, True, attempts=attempts)

    # ── Step 2: fallbacks, in order, each failing soft ───────────────────────
    logger.warning("[fallback] %s: %s extraction is unusable (%s) — trying: %s",
                   pdf_path.name, primary_name, reason, ", ".join(fallback_names))
    for name in fallback_names:
        alt_text: Optional[str] = cache_lookup(name) if cache_lookup else None
        if alt_text is None or looks_garbled(alt_text, n_pages):
            try:
                blocks = make_extractor(name).extract(pdf_path)
                alt_text = blocks[0]["text"] if blocks else ""
            except Exception as exc:
                attempts.append(FallbackAttempt(name, False, f"unavailable: {exc}"))
                logger.warning("[fallback] %s: %s unavailable (%s) — trying next",
                               pdf_path.name, name, exc)
                continue
        bad = looks_garbled(alt_text, n_pages)
        attempts.append(FallbackAttempt(name, bad is None, bad))
        if bad is None:
            logger.warning("[fallback] %s: using %s text for this document "
                           "(the %s output for it is not trustworthy)",
                           pdf_path.name, name, primary_name)
            cache_write(name, pdf_path, alt_text)
            return ExtractionResult(alt_text, name, True, attempts=attempts)
        logger.warning("[fallback] %s: %s also unusable (%s)", pdf_path.name, name, bad)

    return ExtractionResult(text or "", primary_name, False, reason=reason,
                            attempts=attempts)
