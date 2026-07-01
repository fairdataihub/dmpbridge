"""Whole-document Ollama classification on all 10 samples.

Output: data/llmlabeled/sampleN_{model}_whole_doc.json

Usage:
    python run_wholedoc_ollama.py --model llama3.3:70b
    python run_wholedoc_ollama.py --model llama3.1:8b
"""
import argparse
from pathlib import Path

import requests

from dmpbridge import config
from dmpbridge.classifier import SYSTEM_PROMPT, _OUTPUT_SCHEMA
from dmpbridge.logging_setup import get_logger, setup_logging
from dmpbridge.wholedoc import parse_response, run_samples

setup_logging()
logger = get_logger("dmpbridge.run_wholedoc_ollama")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama3.3:70b")
parser.add_argument("--host",  default=config.HOST)
args = parser.parse_args()

MODEL   = args.model
HOST    = args.host.rstrip("/")
TAG     = f"{MODEL.replace(':', '-')}_whole_doc"
PDF_DIR = Path("data/pdfsamples")
OUT_DIR = Path("data/llmlabeled")

logger.info("model=%s  host=%s  tag=%s", MODEL, HOST, TAG)


def _classify(_blocks, payload, prompt, label):
    logger.info("%s sending %d blocks to %s …", label, len(payload), MODEL)
    try:
        resp = requests.post(
            f"{HOST}/api/generate",
            json={
                "model":   MODEL,
                "system":  SYSTEM_PROMPT,
                "prompt":  prompt,
                "stream":  False,
                "format":  _OUTPUT_SCHEMA,
                "options": {"temperature": 0.0, "num_ctx": 32768},
            },
            timeout=600,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("%s ERROR: %s", label, e)
        return []
    raw = resp.json().get("response", "")
    logger.info("%s %d chars", label, len(raw))
    return parse_response(raw, label)


run_samples(PDF_DIR, OUT_DIR, _classify, TAG)
