"""Whole-document strategy — pdfplumber extraction + single LLM call.

Instead of processing blocks in batches, this strategy sends the entire document
to the model in one request.  The model sees the full document at once, which
improves structural continuity (e.g. it never loses track of which section it is
inside) at the cost of higher token usage and longer latency.

Supports Anthropic and Ollama providers.

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
from ..extractor import extract_blocks
from ..logging_setup import get_logger
from ..prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT
from ..wholedoc import apply_labels, build_payload, parse_response
from ..prompt import build_wholedoc_prompt

logger = get_logger(__name__)


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
        self.provider = provider
        self.model    = model

        if provider == "anthropic":
            _key = api_key or config.ANTHROPIC_API_KEY
            if not _key:
                raise ConfigurationError("ANTHROPIC_API_KEY is not set.")
            self._classify_fn = self._make_anthropic_fn(_key)

        elif provider == "ollama":
            _host = host.rstrip("/")
            self._classify_fn = self._make_ollama_fn(_host)

        else:
            raise ConfigurationError(
                f"WholeDocStrategy: unsupported provider {provider!r}. "
                "Choose from: anthropic, ollama"
            )

    # ── Strategy protocol ──────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all blocks in *pdf_path* in one model call.

        1. Extract text blocks with pdfplumber.
        2. Build the whole-document prompt (all blocks in a single payload).
        3. Send to the model; parse the JSON response.
        4. Merge predicted labels back into the extracted block list.
        """
        logger.info("[wholedoc] extracting from %s …", pdf_path.name)
        blocks = extract_blocks(pdf_path)
        if not blocks:
            logger.warning("[wholedoc] no blocks found in %s", pdf_path.name)
            return []

        payload = build_payload(blocks)
        prompt  = build_wholedoc_prompt(payload)

        logger.info("[wholedoc] sending %d blocks to %s / %s …",
                    len(payload), self.provider, self.model)
        parsed  = self._classify_fn(blocks, payload, prompt, pdf_path.stem)
        labeled = apply_labels(blocks, parsed)

        return labeled

    # ── Provider-specific closures ─────────────────────────────────────────────

    def _make_anthropic_fn(self, api_key: str):
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        model  = self.model

        def _call(_blocks, payload, prompt, label):
            resp = client.messages.create(
                model=model,
                max_tokens=16384,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text if resp.content else ""
            logger.info("[wholedoc] %s in=%s out=%s",
                        label, f"{resp.usage.input_tokens:,}", f"{resp.usage.output_tokens:,}")
            return parse_response(raw, label)

        return _call

    def _make_ollama_fn(self, host: str):
        import requests as _req
        model = self.model

        def _call(_blocks, payload, prompt, label):
            try:
                resp = _req.post(
                    f"{host}/api/generate",
                    json={
                        "model":   model,
                        "system":  SYSTEM_PROMPT,
                        "prompt":  prompt,
                        "stream":  False,
                        "format":  OUTPUT_SCHEMA,
                        "options": {"temperature": 0.0, "num_ctx": 32768},
                    },
                    timeout=600,
                )
                resp.raise_for_status()
            except _req.exceptions.RequestException as e:
                logger.error("[wholedoc] %s ERROR: %s", label, e)
                return []
            raw = resp.json().get("response", "")
            logger.info("[wholedoc] %s %d chars", label, len(raw))
            return parse_response(raw, label)

        return _call
