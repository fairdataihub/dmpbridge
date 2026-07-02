"""LLM model backends.

A :class:`ModelBackend` is anything that takes a system prompt and a user
prompt and returns raw text.  Each sub-module contains one concrete backend.

Sub-modules
-----------
ollama      OllamaModel — local Ollama server
anthropic   AnthropicModel — Anthropic messages API
openai      OpenAIModel — OpenAI chat completions API
gemini      GeminiModel — Google Gemini API

Factory
-------
Use :func:`get_model` to get the right backend by provider name instead of
importing each class directly.
"""
from typing import Protocol, runtime_checkable

from ..core import config as _cfg


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal interface every model backend must satisfy.

    A backend is stateless across calls: the same instance can be used for
    multiple :meth:`complete` calls without side effects.
    """

    def complete(self, system: str, prompt: str) -> str:
        """Call the model and return its raw text response.

        Parameters
        ----------
        system:
            System / instruction prompt.
        prompt:
            User-facing prompt (the content to classify).

        Returns
        -------
        str
            Raw model output — may be JSON, markdown-fenced JSON, or prose.
            Callers should pass the result through
            :func:`~dmpbridge.parsers.parse_llm_json`.
        """
        ...


def get_model(
    provider:   str,
    model:      str,
    *,
    host:       str | None = None,
    api_key:    str | None = None,
    max_tokens: int = 4096,
    num_ctx:    int = 32768,
) -> ModelBackend:
    """Return a configured :class:`ModelBackend` for *provider*.

    Parameters
    ----------
    provider:
        ``"ollama"`` | ``"anthropic"`` | ``"openai"`` | ``"gemini"``
    model:
        Model identifier (provider-specific).
    host:
        Ollama base URL — ignored for cloud providers.
    api_key:
        API key — falls back to the matching ``config.*_API_KEY`` value.
    max_tokens:
        Maximum response tokens (Anthropic only; default 4 096).
        Pass 16 384 for whole-document inference.
    num_ctx:
        Ollama context window size (default 32 768).
    """
    p = provider.lower()

    if p == "ollama":
        from .ollama import OllamaModel
        return OllamaModel(
            model=model,
            host=host or _cfg.HOST,
            num_ctx=num_ctx,
        )

    if p == "anthropic":
        from .anthropic import AnthropicModel
        return AnthropicModel(
            model=model,
            api_key=api_key or _cfg.ANTHROPIC_API_KEY,
            max_tokens=max_tokens,
        )

    if p == "openai":
        from .openai import OpenAIModel
        return OpenAIModel(
            model=model,
            api_key=api_key or _cfg.OPENAI_API_KEY,
        )

    if p == "gemini":
        from .gemini import GeminiModel
        return GeminiModel(
            model=model,
            api_key=api_key or _cfg.GEMINI_API_KEY,
        )

    from ..utils import ConfigurationError
    raise ConfigurationError(
        f"Unknown provider {provider!r}. "
        "Choose from: ollama, anthropic, openai, gemini"
    )


__all__ = ["ModelBackend", "get_model"]
