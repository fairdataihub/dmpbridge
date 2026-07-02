"""Batch strategy — pdfplumber extraction + sliding-window batch LLM calls.

This is the default pipeline strategy.  It extracts text blocks with pdfplumber
(one block per visual text line, with bounding boxes and font metadata) and then
sends them to the model in small overlapping batches so each request fits
comfortably within the model's context window.

The underlying classifier supports Ollama, Anthropic, OpenAI, and Gemini via the
shared :func:`~dmpbridge.classifier.get_classifier` factory.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.batch import BatchStrategy

    strategy = BatchStrategy(provider="anthropic", model="claude-opus-4-8")
    blocks   = strategy.run(Path("document.pdf"))
"""
from pathlib import Path

from .. import config
from ..classifier import BATCH_SIZE, CONTEXT_SIZE, get_classifier
from ..preprocess import extract_blocks
from ..logging_setup import get_logger

logger = get_logger(__name__)


class BatchStrategy:
    """Extract blocks with pdfplumber, classify in sliding batches.

    Parameters
    ----------
    provider:
        LLM provider — ``"ollama"`` | ``"anthropic"`` | ``"openai"`` | ``"gemini"``.
    model:
        Model identifier.
    host:
        Ollama base URL (ignored for cloud providers).
    batch_size:
        Number of blocks per LLM request (default: ``config.BATCH_SIZE``).
    context_size:
        Number of already-labeled blocks prepended as sliding context (default: 3).
    """

    def __init__(
        self,
        provider:     str = config.PROVIDER,
        model:        str = config.MODEL,
        host:         str = config.HOST,
        batch_size:   int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        self.provider     = provider
        self.model        = model
        self.batch_size   = batch_size
        self.context_size = context_size
        self._classifier  = get_classifier(
            provider=provider,
            model=model,
            host=host,
            batch_size=batch_size,
            context_size=context_size,
        )

    # ── Strategy protocol ──────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and batch-classify all blocks in *pdf_path*.

        1. Extract text blocks with pdfplumber.
        2. Send them to the LLM in overlapping batches with a sliding context window.
        3. Fill any blocks the LLM skipped with the default label ``answer.text``.
        """
        logger.info("[batch] extracting from %s …", pdf_path.name)
        blocks = extract_blocks(pdf_path)
        if not blocks:
            logger.warning("[batch] no blocks found in %s", pdf_path.name)
            return []

        logger.info("[batch] classifying %d blocks with %s / %s (batch=%d, ctx=%d) …",
                    len(blocks), self.provider, self.model,
                    self.batch_size, self.context_size)
        labeled = self._classifier.classify_blocks(blocks)

        for b in labeled:
            if not b.get("label"):
                b["label"] = "answer.text"

        return labeled
