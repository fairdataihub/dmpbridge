"""Whole-document strategy — pdfplumber extraction + single model call.

All blocks are sent to the model in one request so it sees the full document
at once, improving structural continuity at the cost of higher token usage.

Uses :class:`~dmpbridge.models.ModelBackend` for the model call and
:func:`~dmpbridge.parsers.parse_llm_json` for response parsing — no
strategy-specific closures or duplicated parsing logic.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.wholedoc import WholeDocStrategy

    strategy = WholeDocStrategy(provider="anthropic", model="claude-opus-4-8")
    blocks   = strategy.run(Path("document.pdf"))
"""
import json
from pathlib import Path

from ..core import config
from ..models import get_model
from ..parsers import parse_llm_json
from ..preprocess import extract_blocks
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
        idx = entry.get("id")
        lbl = entry.get("label", "answer.text")
        if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
            result[idx]["label"] = lbl
    for b in result:
        if not b.get("label"):
            b["label"] = "answer.text"
    return result


class WholeDocStrategy:
    """Extract blocks with pdfplumber, classify in a single model call.

    Parameters
    ----------
    provider:
        ``"anthropic"`` or ``"ollama"``.
    model:
        Model identifier.
    host:
        Ollama base URL (ignored for Anthropic).
    api_key:
        Anthropic API key — falls back to ``config.ANTHROPIC_API_KEY``.
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
        api_key:       str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if provider not in ("anthropic", "ollama"):
            raise ConfigurationError(
                f"WholeDocStrategy: unsupported provider {provider!r}. "
                "Choose from: anthropic, ollama"
            )
        self.provider       = provider
        self.model          = model
        self._system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._backend       = get_model(
            provider,
            model,
            host=host,
            api_key=api_key,
            max_tokens=16384,
        )

    # ── Strategy protocol ──────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all blocks in *pdf_path* in one model call.

        1. Extract text blocks with pdfplumber.
        2. Build a single whole-document prompt from all blocks.
        3. Call the model and parse the JSON response.
        4. Merge predicted labels back into the block list.
        """
        logger.info("[wholedoc] extracting from %s …", pdf_path.name)
        blocks = extract_blocks(pdf_path)
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
