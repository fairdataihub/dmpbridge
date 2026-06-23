"""Main pipeline: PDF → pdfplumber extraction → raw JSON → LLM labeling → labeled JSON."""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

from . import config
from .classifier import OllamaClassifier
from .extractor import extract_blocks, save_page_images

# Silence noisy pdfminer internal loggers
for _noisy in ("pdfminer", "pdfminer.pdfpage", "pdfminer.pdfdocument",
               "pdfminer.pdfinterp", "pdfminer.converter", "pdfminer.cmapdb"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_MODEL   = config.MODEL
DEFAULT_HOST    = config.HOST
DEFAULT_RAW_DIR = "data/pdfplumber"


_NUMBERED_HEADING = re.compile(r"^\d+[\.\)]")


def _smooth_labels(blocks: list[dict]) -> list[dict]:
    """
    Fix two systematic LLM errors caused by line-by-line font-style classification:

    1. bold+italic section.title that is NOT a numbered heading is a question opener —
       e.g. "Data Sharing and Data Preservation. A description..." is question.text,
       while "1. Policy and Practice" is a genuine section.title.

    2. italic-only continuation lines following a question.text are mislabeled
       section.description — propagate question.text forward.
    """
    for i, block in enumerate(blocks):
        label     = block.get("label", "")
        is_bold   = block.get("is_bold", False)
        is_italic = block.get("is_italic", False)
        text      = block.get("text", "")

        # Rule 1: bold+italic section.title with no leading number → question.text
        if label == "section.title" and is_bold and is_italic:
            if not _NUMBERED_HEADING.match(text):
                block["label"] = "question.text"
                label = "question.text"

        # Rule 2: italic-only line following question.text mislabeled as section.description
        if label == "section.description" and is_italic and not is_bold and i > 0:
            if blocks[i - 1].get("label") == "question.text":
                block["label"] = "question.text"

    return blocks


def process_pdf(
    pdf_path: Union[str, Path],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    output: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = DEFAULT_RAW_DIR,
    images_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """
    Extract and label all text blocks from a PDF file using an LLM.

    Parameters
    ----------
    pdf_path  : Path to the input PDF.
    model     : Ollama model name (default from config.py).
    host      : Ollama server base URL (default from config.py).
    output    : If given, write the labeled JSON to this path.
    raw_dir   : Folder to save raw pdfplumber extraction JSON before LLM labeling.
                Defaults to "data/pdfplumber". Pass None to skip.
    images_dir: If given, also save per-page PNG images to this folder.

    Returns
    -------
    List of block dicts, each with a 'label' field set to one of:
    title | section.title | section.description | question.text | answer.text
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Step 1: extract ──────────────────────────────────────────────────────
    logger.info(f"Extracting text from {pdf_path.name} …")
    blocks = extract_blocks(pdf_path)
    if not blocks:
        logger.warning("No text blocks found in the PDF.")
        return []

    pages = len({b["page"] for b in blocks})
    logger.info(f"  → {len(blocks)} blocks across {pages} page(s)")

    # ── Step 2: save raw pdfplumber JSON (before labeling) ───────────────────
    if raw_dir is not None:
        raw_path = Path(raw_dir) / f"{pdf_path.stem}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Raw extraction saved → {raw_path}")

    # ── Step 3: optional page images ─────────────────────────────────────────
    if images_dir is not None:
        logger.info(f"Saving page images → {images_dir} …")
        try:
            saved = save_page_images(pdf_path, blocks, output_dir=images_dir)
            logger.info(f"  → {len(saved)} image(s) saved")
        except Exception as exc:
            logger.warning(f"Image export skipped: {exc}")

    # ── Step 4: classify with LLM ────────────────────────────────────────────
    logger.info(f"Classifying with model '{model}' at {host} …")
    clf = OllamaClassifier(model=model, host=host)
    blocks = clf.classify_blocks(blocks)

    # ── Step 5: fill any blocks the LLM did not return a label for ───────────
    for b in blocks:
        if not b.get("label"):
            b["label"] = "answer.text"

    # ── Step 5b: smooth context-dependent mislabels ───────────────────────────
    blocks = _smooth_labels(blocks)

    # ── Step 6: save labeled JSON ─────────────────────────────────────────────
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Labeled JSON saved → {out_path}")

    return blocks
