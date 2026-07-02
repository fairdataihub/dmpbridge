"""Anthropic (Claude) model backend."""
from ..utils import ConfigurationError, get_logger

logger = get_logger(__name__)


class AnthropicModel:
    """Call the Anthropic messages API.

    Parameters
    ----------
    model:
        Anthropic model ID, e.g. ``"claude-opus-4-8"``.
    api_key:
        Anthropic API key.
    max_tokens:
        Maximum tokens in the response.  Use 4 096 for batch calls and
        16 384 for whole-document inference.
    """

    def __init__(
        self,
        model:      str,
        api_key:    str,
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Add it to your .env file:  ANTHROPIC_API_KEY=sk-ant-..."
            )
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is not installed.\n"
                "Install it with:  pip install anthropic"
            )
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = _anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, prompt: str) -> str:
        """Send *system* + *prompt* to Claude and return the raw text response."""
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        logger.info(
            "in=%s  out=%s",
            f"{resp.usage.input_tokens:,}",
            f"{resp.usage.output_tokens:,}",
        )
        return raw
