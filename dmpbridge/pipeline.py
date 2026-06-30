"""Main pipeline: PDF → pdfplumber extraction → raw JSON → LLM labeling → labeled JSON."""

import json
import logging
from pathlib import Path
from typing import Optional, Union

from . import config
from .classifier import OllamaClassifier
from .converter import to_structured
from .extractor import extract_blocks, save_page_images

# Silence noisy pdfminer internal loggers so the terminal output stays clean.
for _noisy in ("pdfminer", "pdfminer.pdfpage", "pdfminer.pdfdocument",
               "pdfminer.pdfinterp", "pdfminer.converter", "pdfminer.cmapdb"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_MODEL   = config.MODEL
DEFAULT_HOST    = config.HOST
DEFAULT_RAW_DIR = "data/pdfplumber"


def process_pdf(
    pdf_path: Union[str, Path],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    output: Optional[Union[str, Path]] = None,
    structured_output: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = DEFAULT_RAW_DIR,
    images_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """Run the full pipeline for one PDF. 1. Extract text blocks with pdfplumber. 2. Optionally save the raw extraction JSON before labeling. 3. Optionally save per-page PNG images with bounding boxes. 4. Send blocks to the LLM for label classification. 5. Fill any blocks the LLM missed with a default label. 6. Save the flat labeled JSON. 7. Optionally convert and save the structured JSON."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1 — extract all text lines from the PDF as individual blocks.
    logger.info(f"Extracting text from {pdf_path.name} …")
    blocks = extract_blocks(pdf_path)
    if not blocks:
        logger.warning("No text blocks found in the PDF.")
        return []

    pages = len({b["page"] for b in blocks})
    logger.info(f"  → {len(blocks)} blocks across {pages} page(s)")

    # Step 2 — save the raw pdfplumber output before any LLM labeling, useful for debugging.
    if raw_dir is not None:
        raw_path = Path(raw_dir) / f"{pdf_path.stem}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Raw extraction saved → {raw_path}")

    # Step 3 — optionally render each page as a PNG with colored bounding boxes per label.
    if images_dir is not None:
        logger.info(f"Saving page images → {images_dir} …")
        try:
            saved = save_page_images(pdf_path, blocks, output_dir=images_dir)
            logger.info(f"  → {len(saved)} image(s) saved")
        except Exception as exc:
            logger.warning(f"Image export skipped: {exc}")

    # Step 4 — send all blocks to the LLM in batches to get label predictions.
    logger.info(f"Classifying with model '{model}' at {host} …")
    clf = OllamaClassifier(model=model, host=host)
    blocks = clf.classify_blocks(blocks)

    # Step 5 — if the LLM skipped any block, default its label to answer.text.
    for b in blocks:
        if not b.get("label"):
            b["label"] = "answer.text"

    # Step 6 — save the flat labeled JSON (one block per line, with label field added).
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Labeled JSON saved → {out_path}")

    # Step 7 — convert the flat block list into the nested DMP Tool JSON schema and save it.
    if structured_output:
        struct_path = Path(structured_output)
        struct_path.parent.mkdir(parents=True, exist_ok=True)
        struct_path.write_text(
            json.dumps(to_structured(blocks), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Structured JSON saved → {struct_path}")

    return blocks
