"""Vision-batch strategy — pdfplumber extraction + page image sent to Claude.

For each page:
  1. Extract text blocks with pdfplumber (text, position, font metadata).
  2. Render the page as a PNG image (base64-encoded).
  3. Send both the image and the block list to Claude in a single multimodal call.

Claude uses the visual layout (font sizes, spacing, bold/italic rendering) alongside
the structured text to classify each block.  This gives more context than text alone
without the paragraph-level coarseness of the pdf-direct strategy.

Only the Anthropic provider is supported because this strategy relies on Claude's
vision capability.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.vision_batch import VisionBatchStrategy

    strategy = VisionBatchStrategy(model="claude-opus-4-8")
    blocks   = strategy.run(Path("document.pdf"))
"""
import base64
import json
from pathlib import Path

from ..core import config
from ..parsers import parse_llm_json
from ..preprocess import extract_blocks, render_pages
from ..prompts import LABELS, SYSTEM_PROMPT
from ..utils import ConfigurationError, get_logger

logger = get_logger(__name__)

_PAGE_PROMPT_TEMPLATE = (
    "Page {page_num} of the DMP document is shown in the image above.\n\n"
    "The image shows the visual layout — use it to identify headings, body text, "
    "font size hierarchy, and spacing.\n\n"
    "Below are the text blocks already extracted from this page. "
    "Classify each block using BOTH the visual context and the text.\n\n"
    "BLOCKS TO CLASSIFY — return a JSON array with exactly {n} entries, "
    "one per block, preserving the same id values:\n"
    "{payload}"
)


def _file_to_b64(path: Path) -> str:
    """Read a PNG file from disk and return its base64 encoding."""
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _classify_page(
    client,
    model: str,
    page_num: int,
    page_b64: str,
    payload: list[dict],
    max_tokens: int,
) -> list[dict]:
    """Send one page image + block list to Claude and return the parsed labels."""
    prompt_text = _PAGE_PROMPT_TEMPLATE.format(
        page_num=page_num,
        n=len(payload),
        payload=json.dumps(payload, ensure_ascii=False),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       page_b64,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
    )

    raw = response.content[0].text if response.content else ""
    logger.info(
        "  page %d — in=%s  out=%s",
        page_num,
        f"{response.usage.input_tokens:,}",
        f"{response.usage.output_tokens:,}",
    )
    return parse_llm_json(raw, label=f"page={page_num}")


class VisionBatchStrategy:
    """Extract blocks with pdfplumber, classify page-by-page with image context.

    Each page is rendered as a PNG and sent to Claude alongside the text blocks
    extracted from that page.  Claude uses the visual layout to classify each block.

    Parameters
    ----------
    model:
        Anthropic model that supports vision (e.g. ``"claude-opus-4-8"``).
    api_key:
        Anthropic API key — falls back to ``config.ANTHROPIC_API_KEY``.
    images_dir:
        Root directory where page PNGs are saved, organised as
        ``{images_dir}/{pdf_stem}/page_001.png``.
        Defaults to ``data/output/page_images``.
        Pages already on disk are reused — rendering only runs once per PDF
        regardless of how many models or experiments consume the images.
    resolution:
        DPI for page image rendering (default: 150).
    max_tokens:
        Maximum response tokens per page call (default: 4096).
    """

    def __init__(
        self,
        model:      str = "claude-opus-4-8",
        api_key:    str | None = None,
        images_dir: str | Path = "data/output/page_images",
        resolution: int = 150,
        max_tokens: int = 4096,
    ) -> None:
        _key = api_key or config.ANTHROPIC_API_KEY
        if not _key:
            raise ConfigurationError(
                "VisionBatchStrategy requires ANTHROPIC_API_KEY to be set."
            )
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is not installed.\n"
                "Install it with:  pip install anthropic"
            )
        self.model       = model
        self.images_dir  = Path(images_dir)
        self.resolution  = resolution
        self.max_tokens  = max_tokens
        self._client     = anthropic.Anthropic(api_key=_key)

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all blocks in *pdf_path* using page images.

        1. Extract all text blocks from the PDF via pdfplumber.
        2. For each page, render a PNG and call Claude with image + blocks.
        3. Merge predicted labels back into the block list.
        """
        logger.info("[vision_batch] extracting from %s …", pdf_path.name)
        blocks = extract_blocks(pdf_path)
        if not blocks:
            logger.warning("[vision_batch] no blocks found in %s", pdf_path.name)
            return []

        result = [dict(b) for b in blocks]
        pages  = sorted(set(b["page"] for b in result))

        # Render pages to disk once; subsequent runs (and other models) reuse them.
        img_dir   = self.images_dir / pdf_path.stem
        page_imgs = render_pages(pdf_path, img_dir, resolution=self.resolution)

        logger.info(
            "[vision_batch] %d blocks across %d pages — model=%s  images=%s",
            len(result), len(pages), self.model, img_dir,
        )

        for page_num in pages:
            global_idx  = [i for i, b in enumerate(result) if b["page"] == page_num]
            page_blocks = [result[i] for i in global_idx]
            if not page_blocks:
                continue

            img_b64 = _file_to_b64(page_imgs[page_num - 1])
            payload = [
                {
                    "id":     j,
                    "text":   b["text"],
                    "bold":   b["is_bold"],
                    "italic": b.get("is_italic", False),
                }
                for j, b in enumerate(page_blocks)
            ]

            logger.info("  page %d — %d blocks", page_num, len(page_blocks))
            parsed = _classify_page(
                self._client, self.model, page_num,
                img_b64, payload, self.max_tokens,
            )

            for entry in parsed:
                local_idx = entry.get("id")
                lbl       = entry.get("label", "answer.text")
                if local_idx is not None and 0 <= local_idx < len(global_idx):
                    if lbl in LABELS:
                        result[global_idx[local_idx]]["label"] = lbl

        for b in result:
            if not b.get("label"):
                b["label"] = "answer.text"

        return result
