"""Whole-document strategy — configurable extraction + single model call.

All blocks are sent to the model in one request so it sees the full document
at once, improving structural continuity at the cost of higher token usage.

The extraction step is handled by a :class:`~dmpbridge.extractors.BaseExtractor`
so the pipeline can compare pdfplumber, Docling, and LightOnOCR side-by-side
without changing any other code.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.wholedoc import WholeDocStrategy

    strategy = WholeDocStrategy(model="llama3.3:70b", extractor="docling")
    blocks   = strategy.run(Path("document.pdf"))
"""
import json
from pathlib import Path

from ..core import config
from ..extractors import get_extractor
from ..models import get_model
from ..parsers import parse_llm_json
from ..prompts import LABELS, SYSTEM_PROMPT
from ..utils import ConfigurationError, get_logger

logger = get_logger(__name__)


def _build_wholedoc_prompt(payload: list[dict]) -> str:
    return (
        f"CLASSIFY ALL BLOCKS — return a JSON array with exactly {len(payload)} entries, "
        f"one per block, in the same order:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _build_payload(blocks: list[dict]) -> list[dict]:
    return [
        {
            "id":     j,
            "text":   b["text"],
            "bold":   b["is_bold"],
            "italic": b.get("is_italic", False),
            "page":   b["page"],
        }
        for j, b in enumerate(blocks)
    ]


def _apply_labels(blocks: list[dict], parsed: list[dict]) -> list[dict]:
    result = [dict(b) for b in blocks]
    for entry in parsed:
        idx  = entry.get("id")
        lbl  = entry.get("label", "answer.text")
        conf = float(entry.get("confidence", 1.0))
        if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
            result[idx]["label"]      = lbl
            result[idx]["confidence"] = max(0.0, min(1.0, conf))
    for b in result:
        if not b.get("label"):
            b["label"] = "answer.text"
        if "confidence" not in b:
            b["confidence"] = 1.0
    return result


class WholeDocStrategy:
    """Extract blocks from a PDF then classify them in a single model call.

    The extraction step is delegated to a :class:`~dmpbridge.extractors.BaseExtractor`
    so that pdfplumber, Docling, and LightOnOCR can be swapped in without
    touching the prompt or evaluation code.

    Parameters
    ----------
    provider:
        LLM provider — only ``"ollama"`` is supported.
    model:
        Ollama model identifier (e.g. ``"llama3.3:70b"``).
    host:
        Ollama base URL.
    extractor:
        PDF extraction backend — ``"pdfplumber"`` (default), ``"docling"``,
        or ``"lighton"``.
    system_prompt:
        Override the default system prompt.  Used by the rotation evaluation
        design to inject dynamic few-shot examples from a specific sample pair.
        When ``None``, the module-level ``SYSTEM_PROMPT`` constant is used.
    """

    def __init__(
        self,
        provider:      str = config.PROVIDER,
        model:         str = config.MODEL,
        host:          str = config.HOST,
        extractor:     str = "pdfplumber",
        system_prompt: str | None = None,
    ) -> None:
        if provider != "ollama":
            raise ConfigurationError(
                f"WholeDocStrategy: unsupported provider {provider!r}. Only 'ollama' is supported."
            )
        self.provider       = provider
        self.model          = model
        self._system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._extractor     = get_extractor(extractor)
        self._backend       = get_model(
            provider,
            model,
            host=host,
            max_tokens=16384,
        )

    # ── Strategy protocol ──────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all blocks in *pdf_path* in one model call."""
        logger.info("[wholedoc] extracting from %s via %s …",
                    pdf_path.name, type(self._extractor).__name__)
        blocks = self._extractor.extract(pdf_path)
        if not blocks:
            logger.warning("[wholedoc] no blocks found in %s", pdf_path.name)
            return []

        payload = _build_payload(blocks)
        prompt  = _build_wholedoc_prompt(payload)

        logger.info("[wholedoc] sending %d blocks to %s / %s …",
                    len(payload), self.provider, self.model)
        raw    = self._backend.complete(self._system_prompt, prompt)
        parsed = parse_llm_json(raw, label=pdf_path.stem)
        return _apply_labels(blocks, parsed)
