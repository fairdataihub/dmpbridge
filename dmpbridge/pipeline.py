"""Main pipeline: PDF → pdfplumber extraction → raw JSON → LLM labeling → labeled JSON."""

# What this file does — step by step:
#   Step 1 — open the PDF and extract every text line as a block (text, position, font info)
#   Step 2 — save the raw extraction to disk before the LLM touches anything (useful for debugging)
#   Step 3 — send all blocks to the LLM in batches to get a label for each block
#   Step 4 — any block the LLM skipped gets a default label of "answer.text"
#   Step 5 — save the flat labeled JSON  (one block per entry, label field now filled in)
#   Step 6 — convert the flat list into the nested DMP Tool schema (section → question → answer)
#   Step 7 — save the structured JSON to disk

import json
from pathlib import Path
from typing import Optional, Union

from . import config
from .classifier import get_classifier
from .converter import to_structured
from .exceptions import ExtractionError
from .extractor import extract_blocks, save_page_images
from .logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_PROVIDER = config.PROVIDER
DEFAULT_MODEL    = config.MODEL
DEFAULT_HOST     = config.HOST
DEFAULT_RAW_DIR  = "data/pdfplumber"


def process_pdf(
    pdf_path: Union[str, Path],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    output: Optional[Union[str, Path]] = None,
    structured_output: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = DEFAULT_RAW_DIR,
    images_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """Run the full pipeline for one PDF.

    1. Extract text blocks with pdfplumber.
    2. Optionally save the raw extraction JSON before labeling.
    3. Optionally save per-page PNG images with bounding boxes.
    4. Send blocks to the LLM for label classification.
    5. Fill any blocks the LLM missed with a default label.
    6. Save the flat labeled JSON.
    7. Optionally convert and save the structured JSON.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError("PDF not found: %s" % pdf_path)

    # Step 1 — extract all text lines from the PDF as individual blocks.
    logger.info("Extracting text from %s …", pdf_path.name)
    blocks = extract_blocks(pdf_path)
    if not blocks:
        logger.warning("No text blocks found in the PDF.")
        return []

    pages = len({b["page"] for b in blocks})
    logger.info("  → %d blocks across %d page(s)", len(blocks), pages)

    # Step 2 — save the raw pdfplumber output before any LLM labeling, useful for debugging.
    if raw_dir is not None:
        raw_path = Path(raw_dir) / f"{pdf_path.stem}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Raw extraction saved → %s", raw_path)

    # Step 3 — optionally render each page as a PNG with colored bounding boxes per label.
    if images_dir is not None:
        logger.info("Saving page images → %s …", images_dir)
        try:
            saved = save_page_images(pdf_path, blocks, output_dir=images_dir)
            logger.info("  → %d image(s) saved", len(saved))
        except ExtractionError as exc:
            logger.warning("Image export skipped: %s", exc)

    # Step 4 — send all blocks to the LLM in batches to get label predictions.
    logger.info("Classifying with %s / %s …", provider, model)
    clf = get_classifier(provider=provider, model=model, host=host)
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
        logger.info("Labeled JSON saved → %s", out_path)

    # Step 7 — convert the flat block list into the nested DMP Tool JSON schema and save it.
    if structured_output:
        struct_path = Path(structured_output)
        struct_path.parent.mkdir(parents=True, exist_ok=True)
        struct_path.write_text(
            json.dumps(to_structured(blocks), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Structured JSON saved → %s", struct_path)

    return blocks
