"""Batch strategy — pdfplumber extraction + sliding-window batch LLM calls.

This is the default pipeline strategy.  It extracts text blocks with pdfplumber
(one block per visual text line, with bounding boxes and font metadata) and then
sends them to the model in small overlapping batches so each request fits
comfortably within the model's context window.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.batch import BatchStrategy

    strategy = BatchStrategy(provider="anthropic", model="claude-opus-4-8")
    blocks   = strategy.run(Path("document.pdf"))
"""
import json
from pathlib import Path

from ..core import config
from ..models import get_model
from ..parsers import parse_llm_json
from ..preprocess import extract_blocks
from ..prompts import LABELS, SYSTEM_PROMPT
from ..utils import get_logger

logger = get_logger(__name__)


def _build_batch_prompt(
    batch: list[dict],
    offset: int,
    context: list[dict] | None = None,
) -> str:
    ctx_section = ""
    if context:
        ctx_blocks = [
            {
                "text":   b["text"],
                "bold":   b["is_bold"],
                "italic": b.get("is_italic", False),
                "label":  b.get("label", "?"),
            }
            for b in context
        ]
        ctx_section = (
            "PRECEDING BLOCKS (already labeled — use for context only, "
            "do not output labels for these):\n"
            + json.dumps(ctx_blocks, ensure_ascii=False)
            + "\n\n"
        )

    payload = [
        {
            "id":     offset + j,
            "text":   b["text"],
            "bold":   b["is_bold"],
            "italic": b.get("is_italic", False),
            "page":   b["page"],
        }
        for j, b in enumerate(batch)
    ]

    return (
        ctx_section
        + "TO CLASSIFY — return a JSON array with exactly %d entries, one per block:\n"
        % len(batch)
        + json.dumps(payload, ensure_ascii=False)
    )

BATCH_SIZE   = config.BATCH_SIZE
CONTEXT_SIZE = 3


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
    system_prompt:
        Override the default system prompt.  Used by the rotation evaluation design
        to inject dynamic few-shot examples drawn from a specific sample pair.
        When ``None``, the module-level ``SYSTEM_PROMPT`` constant is used.
    """

    def __init__(
        self,
        provider:      str = config.PROVIDER,
        model:         str = config.MODEL,
        host:          str = config.HOST,
        batch_size:    int = BATCH_SIZE,
        context_size:  int = CONTEXT_SIZE,
        system_prompt: str | None = None,
    ) -> None:
        self.provider       = provider
        self.model          = model
        self.batch_size     = batch_size
        self.context_size   = context_size
        self._system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._backend       = get_model(provider, model, host=host)

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
        result = self._classify_blocks(blocks)

        for b in result:
            if not b.get("label"):
                b["label"] = "answer.text"
            if "confidence" not in b:
                b["confidence"] = 1.0

        return result

    # ── Internal ───────────────────────────────────────────────────────────────

    def _classify_blocks(self, blocks: list[dict]) -> list[dict]:
        result = [dict(b) for b in blocks]

        for start in range(0, len(result), self.batch_size):
            batch   = result[start : start + self.batch_size]
            context = result[max(0, start - self.context_size) : start]

            logger.info("  Classifying blocks %d–%d …", start, start + len(batch) - 1)

            prompt = _build_batch_prompt(batch, start, context)
            raw    = self._backend.complete(self._system_prompt, prompt)
            labels = parse_llm_json(raw, label=f"offset={start}")

            for entry in labels:
                idx  = entry.get("id")
                lbl  = entry.get("label", "answer.text")
                conf = float(entry.get("confidence", 1.0))
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"]      = lbl
                    result[idx]["confidence"] = max(0.0, min(1.0, conf))

        return result
