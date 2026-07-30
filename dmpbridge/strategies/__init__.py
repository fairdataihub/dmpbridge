"""Strategy protocol and factory for the DMPBridge extraction pipeline.

A Strategy encapsulates one end-to-end approach for turning a DMP PDF into
a flat list of labeled text blocks.  Each strategy owns its own preprocessing
and model-call logic, so the pipeline can swap strategies without changing any
other code.

Available strategies
--------------------
WholeDocStrategy — pdfplumber extraction → single LLM call for the whole doc

Usage
-----
    from dmpbridge.strategies import get_strategy, Strategy

    strategy = get_strategy("wholedoc", model="llama3.3:70b")
    blocks   = strategy.run(Path("document.pdf"))
"""
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Strategy(Protocol):
    """Common interface for all extraction strategies.

    A strategy takes a PDF path and returns a flat list of labeled blocks::

        [{"text": "...", "label": "answer.text", "page": 1, ...}, ...]

    Each strategy is configured at construction time (model, provider, etc.)
    and is stateless across calls to :meth:`run`.
    """

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all text blocks from *pdf_path*.

        Parameters
        ----------
        pdf_path:
            Path to an existing PDF file.

        Returns
        -------
        list[dict]
            Flat list of block dicts, each with at least ``text``, ``label``,
            and ``page`` keys.
        """
        ...


def get_strategy(
    name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    host: str | None = None,
    extractor: str = "pdfplumber",
    system_prompt: str | None = None,
) -> Strategy:
    """Return a configured Strategy instance by name.

    Parameters
    ----------
    name:
        ``"wholedoc"`` — the only supported strategy.
    provider:
        LLM provider — only ``"ollama"`` is supported.
        Falls back to ``config.PROVIDER`` when omitted.
    model:
        Model identifier (e.g. ``"llama3.3:70b"``).
        Falls back to ``config.MODEL`` when omitted.
    host:
        Ollama base URL.  Falls back to ``config.HOST`` when omitted.
    extractor:
        PDF extraction backend — ``"pdfplumber"`` (default), ``"docling"``,
        or ``"lighton"``.
    """
    from ..core import config as _cfg
    from .wholedoc import WholeDocStrategy

    _provider = provider or _cfg.PROVIDER
    _model    = model    or _cfg.MODEL
    _host     = host     or _cfg.HOST

    if name == "wholedoc":
        kwargs = {"extractor": extractor}
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        return WholeDocStrategy(provider=_provider, model=_model, host=_host, **kwargs)

    raise ValueError(
        f"Unknown strategy {name!r}. Only 'wholedoc' is supported."
    )


__all__ = ["Strategy", "get_strategy"]
