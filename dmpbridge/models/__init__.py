"""LLM model backends.

A :class:`ModelBackend` is anything that takes a system prompt and a user
prompt and returns raw text.  Each sub-module contains one concrete backend.

Sub-modules
-----------
ollama      OllamaModel — local Ollama server

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

    def complete(self, system: str, prompt: str, *, schema: dict) -> str:
        """Call the model and return its raw text response.

        Parameters
        ----------
        system:
            System / instruction prompt.
        prompt:
            User-facing prompt (the content to classify).
        schema:
            Structured-output schema the response must conform to.

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
        ``"ollama"`` — only supported provider.
    model:
        Model identifier (e.g. ``"llama3.3:70b"``).
    host:
        Ollama base URL.
    api_key:
        Unused — kept for interface compatibility.
    max_tokens:
        Unused — kept for interface compatibility.
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

    from ..utils import ConfigurationError
    raise ConfigurationError(
        f"Unknown provider {provider!r}. Only 'ollama' is supported."
    )


__all__ = ["ModelBackend", "get_model"]
