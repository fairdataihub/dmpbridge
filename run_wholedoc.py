"""Whole-document Claude classification on all 10 samples.

Output: data/llmlabeled/sampleN_claude-opus-4-8_whole_doc.json

Usage: python run_wholedoc.py
"""
from pathlib import Path

import anthropic

from dmpbridge import config
from dmpbridge.classifier import SYSTEM_PROMPT
from dmpbridge.logging_setup import get_logger, setup_logging
from dmpbridge.wholedoc import parse_response, run_samples

setup_logging()
logger = get_logger("dmpbridge.run_wholedoc")

MODEL   = "claude-opus-4-8"
TAG     = f"{MODEL}_whole_doc"
PDF_DIR = Path("data/pdfsamples")
OUT_DIR = Path("data/llmlabeled")

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _classify(_blocks, payload, prompt, label):
    logger.info("%s sending %d blocks to %s …", label, len(payload), MODEL)
    resp = _client.messages.create(
        model=MODEL,
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


run_samples(PDF_DIR, OUT_DIR, _classify, TAG)
