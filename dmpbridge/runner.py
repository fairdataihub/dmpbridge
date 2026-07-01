"""Whole-document inference CLI — supports Anthropic and Ollama providers.

Usage:
    dmpbridge-wholedoc                          # uses provider + model from config
    dmpbridge-wholedoc --provider ollama --model llama3.3:70b
    dmpbridge-wholedoc --provider anthropic --model claude-opus-4-8
"""
import argparse
from pathlib import Path

from . import config
from .exceptions import ConfigurationError
from .logging_setup import get_logger, setup_logging
from .prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT
from .wholedoc import parse_response, run_samples

logger = get_logger(__name__)


def _make_anthropic_classify(model: str, api_key: str):
    """Return a classify_fn that calls the Anthropic API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def _classify(_blocks, payload, prompt, label):
        logger.info("%s sending %d blocks to %s …", label, len(payload), model)
        resp = client.messages.create(
            model=model,
            max_tokens=16384,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""
        logger.info(
            "%s in=%s  out=%s",
            label,
            f"{resp.usage.input_tokens:,}",
            f"{resp.usage.output_tokens:,}",
        )
        return parse_response(raw, label)

    return _classify


def _make_ollama_classify(model: str, host: str):
    """Return a classify_fn that calls a local Ollama server."""
    import requests as _requests

    def _classify(_blocks, payload, prompt, label):
        logger.info("%s sending %d blocks to %s …", label, len(payload), model)
        try:
            resp = _requests.post(
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
        except _requests.exceptions.RequestException as e:
            logger.error("%s ERROR: %s", label, e)
            return []
        raw = resp.json().get("response", "")
        logger.info("%s %d chars", label, len(raw))
        return parse_response(raw, label)

    return _classify


def run_wholedoc(
    provider: str,
    model: str,
    pdf_dir: Path,
    out_dir: Path,
    sample_range: range = range(1, 11),
    host: str | None = None,
) -> None:
    """Run whole-document inference for all samples using the given provider."""
    tag = f"{model.replace(':', '-')}_whole_doc"

    if provider == "anthropic":
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set.")
        classify_fn = _make_anthropic_classify(model, api_key)
    elif provider == "ollama":
        _host = (host or config.HOST).rstrip("/")
        logger.info("model=%s  host=%s  tag=%s", model, _host, tag)
        classify_fn = _make_ollama_classify(model, _host)
    else:
        raise ConfigurationError(
            "Unknown provider %r. Choose from: anthropic, ollama" % provider
        )

    run_samples(pdf_dir, out_dir, classify_fn, tag, sample_range)


def main() -> None:
    """CLI entry point: dmpbridge-wholedoc."""
    setup_logging()
    ap = argparse.ArgumentParser(
        description="Run whole-document LLM classification on DMP PDF samples."
    )
    ap.add_argument(
        "--provider", default=config.PROVIDER, choices=["anthropic", "ollama"],
        help="LLM provider (default: %(default)s)",
    )
    ap.add_argument("--model",   default=config.MODEL, help="Model name (default: %(default)s)")
    ap.add_argument("--host",    default=config.HOST,  help="Ollama host URL (default: %(default)s)")
    ap.add_argument("--pdf-dir", default="data/pdfsamples", type=Path)
    ap.add_argument("--out-dir", default="data/llmlabeled",  type=Path)
    ap.add_argument("--start",   default=1,  type=int, help="First sample index (inclusive)")
    ap.add_argument("--end",     default=10, type=int, help="Last sample index (inclusive)")
    args = ap.parse_args()

    run_wholedoc(
        provider=args.provider,
        model=args.model,
        pdf_dir=args.pdf_dir,
        out_dir=args.out_dir,
        sample_range=range(args.start, args.end + 1),
        host=args.host,
    )


if __name__ == "__main__":
    main()
