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
from pathlib import Path

from .. import config
from ..exceptions import ConfigurationError
from ..logging_setup import get_logger
from ..models import get_model
from ..parsers import parse_llm_json
from ..preprocess import extract_blocks
from ..prompt import LABELS, SYSTEM_PROMPT, build_wholedoc_prompt

logger = get_logger(__name__)


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
    """

    def __init__(
        self,
        provider: str = config.PROVIDER,
        model:    str = config.MODEL,
        host:     str = config.HOST,
        api_key:  str | None = None,
    ) -> None:
        if provider not in ("anthropic", "ollama"):
            raise ConfigurationError(
                f"WholeDocStrategy: unsupported provider {provider!r}. "
                "Choose from: anthropic, ollama"
            )
        self.provider = provider
        self.model    = model
        self._backend = get_model(
            provider,
            model,
            host=host,
            api_key=api_key,
            max_tokens=16384,   # whole-doc responses are large
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
        prompt  = build_wholedoc_prompt(payload)

        logger.info("[wholedoc] sending %d blocks to %s / %s …",
                    len(payload), self.provider, self.model)
        raw    = self._backend.complete(SYSTEM_PROMPT, prompt)
        parsed = parse_llm_json(raw, label=pdf_path.stem)
        return _apply_labels(blocks, parsed)
