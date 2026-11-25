"""One uploaded PDF through all six stages, in memory.

The same steps as ``dmpbridge-pipeline``, with three differences for a web server:

- Nothing is written under ``data/output``. The command line caches every stage by
  file name (``sample1.json``), so two uploads would read each other's results.
- A PDF whose text can't be read is an error for the user, not a warning: the model
  would make up a plan from unreadable text.
- Each model step is checked against the shape the web app reads (see :mod:`.shape`)
  and run again, up to ``attempts`` times, when its answer is not valid JSON, does not
  fit, or takes longer than ``OllamaModel.timeout``. The model's answer can differ from
  one run to the next, so a second try usually works.
"""
from pathlib import Path
from typing import Callable, Optional

import requests

from ..core.converter import to_structured
from ..core.pipeline import normalize_fallbacks
from ..evaluation.annotation_rules import apply_new_annotation_rules
from ..extractors import get_extractor
from ..extractors.fallback import extract_with_fallback
from ..rda.extract import RdaExtractionError, extract_rda, merge
from ..strategies.wholedoc import WholeDocStrategy
from ..utils import ProviderConnectionError, get_logger
from .jobs import JobFailed
from .shape import check_narrative, check_rda

logger = get_logger(__name__)

UNREADABLE = ("The text in this PDF can't be read. It may be a scanned document. "
              "Please upload a PDF with selectable text.")
NO_MODEL = "The AI model server isn't reachable right now. Please try again later."
NO_NARRATIVE = "The sections and questions couldn't be extracted from this plan. Please try again."
NO_RDA = "The metadata couldn't be extracted from this plan. Please try again."


class _DoesNotFit(Exception):
    """The model answered, but not in a shape the web app can read."""


def _retry(step: Callable[[], dict], attempts: int, what: str, failed: str) -> dict:
    """Run *step* until it gives a usable answer, at most *attempts* times."""
    for attempt in range(1, attempts + 1):
        try:
            return step()
        # A timed-out answer is retried too: a run that loops one time usually doesn't the next.
        except (RdaExtractionError, _DoesNotFit, requests.exceptions.ReadTimeout) as e:
            logger.warning("[api] %s, attempt %d of %d: %s", what, attempt, attempts, e)
    raise JobFailed(failed)


def make_runner(narrative_model: str, rda_model: str, host: str,
                extractor: str = "pdfplumber", fallback: Optional[str] = "auto",
                attempts: int = 3):
    """A runner for :class:`~dmpbridge.api.jobs.JobQueue` with these settings."""
    fallbacks = normalize_fallbacks(fallback, extractor) or []

    def run(pdf_path: Path, set_stage: Callable[[str], None]) -> dict:
        # Stage 1: the plan's text, with the OCR fallback for a broken text layer.
        # No cache in either direction: the stage-1 cache is keyed by file name.
        set_stage("reading")
        read = extract_with_fallback(pdf_path, get_extractor(extractor), fallbacks,
                                     cache_write=lambda *_: None)
        if not read.ok or not read.text.strip():
            raise JobFailed(UNREADABLE)

        def narrative() -> dict:
            # Stages 2-4: label each block, nest them, apply the rules.
            blocks = strategy.classify_entire_document(read.text)
            doc = apply_new_annotation_rules(to_structured(blocks))
            problem = check_narrative(doc)
            if problem:
                raise _DoesNotFit(problem)
            return doc

        def rda() -> dict:
            # Stage 5: the RDA JSON from the same text.
            doc, _ = extract_rda(read.text, rda_model, host=host)
            problem = check_rda(doc)
            if problem:
                raise _DoesNotFit(problem)
            return doc

        try:
            set_stage("narrative")
            strategy = WholeDocStrategy(model=narrative_model, host=host, extractor=extractor)
            narrative_doc = _retry(narrative, attempts, "narrative", NO_NARRATIVE)
            set_stage("rda")
            rda_doc = _retry(rda, attempts, "RDA", NO_RDA)
        except (ProviderConnectionError, requests.exceptions.ConnectionError) as e:
            logger.warning("[api] %s", e)
            raise JobFailed(NO_MODEL) from None

        # Stage 6.
        return merge(narrative_doc, rda_doc)

    return run
