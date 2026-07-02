"""PDF-direct classification — send raw PDF bytes to Claude.

:func:`classify_pdf` is the core function: it base64-encodes a PDF, sends it
to Claude's document API, and returns a list of labeled paragraph-level blocks.

The CLI (``dmpbridge-pdf``) runs this over a batch of sample PDFs using
:class:`~dmpbridge.strategies.pdf_direct.PdfDirectStrategy`.

Usage:
    dmpbridge-pdf
    dmpbridge-pdf --model claude-opus-4-8 --start 1 --end 10
"""
import argparse
import base64
import json  # still used for json.dumps in main()
from pathlib import Path

from . import config
from .exceptions import ConfigurationError
from .logging_setup import get_logger, setup_logging
from .parsers import parse_llm_json
from .prompt import LABELS, SYSTEM_PROMPT

logger = get_logger(__name__)

_USER_PROMPT = (
    "Read this DMP (Data Management Plan) document and classify every text block you see.\n\n"
    "Return a JSON array — one entry per block — in document reading order:\n"
    '  [{"text": "<exact text of the block>", "label": "<label>", "page": <page number>}, ...]\n\n'
    "Rules:\n"
    "- Include EVERY paragraph, heading, title, question, and answer block.\n"
    "- Do not merge or split blocks — one logical text unit = one entry.\n"
    "- Use only the 5 labels defined in the system prompt.\n"
    "- Return only the JSON array. No explanation, no markdown fences."
)


def classify_pdf(pdf_path: Path, model: str, api_key: str) -> list[dict]:
    """Send a PDF directly to Claude and return a list of labeled text blocks.

    Each block is a dict with keys: ``text``, ``label``, ``page``.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file to classify.
    model:
        Anthropic model that supports the PDF document API.
    api_key:
        Anthropic API key.
    """
    import anthropic

    client  = anthropic.Anthropic(api_key=api_key)
    pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type":       "base64",
                            "media_type": "application/pdf",
                            "data":       pdf_b64,
                        },
                    },
                    {"type": "text", "text": _USER_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text if response.content else ""
    logger.info(
        "in=%s  out=%s  raw=%d chars",
        f"{response.usage.input_tokens:,}",
        f"{response.usage.output_tokens:,}",
        len(raw),
    )

    parsed = parse_llm_json(raw, label=pdf_path.name)

    valid = []
    for b in parsed:
        if not isinstance(b, dict) or not b.get("text", "").strip():
            continue
        if b.get("label") not in LABELS:
            b["label"] = "answer.text"
        valid.append(b)

    return valid


def main() -> None:
    """CLI entry point: dmpbridge-pdf."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Classify DMP PDFs by sending them directly to Claude (no pdfplumber)."
    )
    ap.add_argument("--model",   default="claude-opus-4-8",
                    help="Anthropic model to use (default: %(default)s)")
    ap.add_argument("--pdf-dir", default="data/pdfsamples", type=Path)
    ap.add_argument("--out-dir", default="data/llmlabeled",  type=Path)
    ap.add_argument("--start",   default=1,  type=int, help="First sample index (inclusive)")
    ap.add_argument("--end",     default=10, type=int, help="Last sample index (inclusive)")
    args = ap.parse_args()

    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not set.")

    from .strategies.pdf_direct import PdfDirectStrategy
    strategy = PdfDirectStrategy(model=args.model, api_key=api_key)

    tag     = f"{args.model}_pdf"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("model=%s  tag=%s  samples=%d–%d", args.model, tag, args.start, args.end)

    for i in range(args.start, args.end + 1):
        label    = f"[sample{i}]"
        pdf_path = args.pdf_dir / f"sample{i}.pdf"
        out_path = out_dir / f"sample{i}_{tag}.json"

        if out_path.exists():
            logger.info("%s already exists — skipping", label)
            continue

        if not pdf_path.exists():
            logger.warning("%s PDF not found: %s", label, pdf_path)
            continue

        logger.info("%s sending %s to %s …", label, pdf_path.name, args.model)
        blocks = strategy.run(pdf_path)
        logger.info("%s %d blocks returned", label, len(blocks))

        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("%s saved → %s", label, out_path.name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
