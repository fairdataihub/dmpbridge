"""Vision-batch strategy — pdfplumber extraction + page image sent to a vision LLM.

For each page:
  1. Extract text blocks with pdfplumber (text, position, font metadata).
  2. Render the page as a PNG image (base64-encoded, saved to disk).
  3. Send both the image and the block list to the model in a single multimodal call.

The model uses the visual layout (font sizes, spacing, bold/italic rendering) alongside
the structured text to classify each block.  This gives more context than text alone
without the paragraph-level coarseness of the pdf-direct strategy.

Supported providers
-------------------
- ``"anthropic"`` — Claude vision models (e.g. ``claude-opus-4-8``)
- ``"ollama"``    — Local Ollama vision models (e.g. ``qwen2-vl:7b``, ``llama3.2-vision:11b``)

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.vision_batch import VisionBatchStrategy

    # Anthropic
    strategy = VisionBatchStrategy(model="claude-opus-4-8", provider="anthropic")
    blocks   = strategy.run(Path("document.pdf"))

    # Ollama (local)
    strategy = VisionBatchStrategy(model="qwen2-vl:7b", provider="ollama")
    blocks   = strategy.run(Path("document.pdf"))
"""
import base64
import json
from pathlib import Path

import requests as _requests

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

# Ollama-specific prompt: plainer format with a concrete output example to
# prevent the model from echoing back the input instead of adding labels.
_OLLAMA_PROMPT_TEMPLATE = (
    "Page {page_num} of a Data Management Plan is shown in the image above.\n\n"
    "Use the image to see font sizes, bold/italic formatting, and layout.\n\n"
    "TASK: Assign exactly ONE label to each text block listed below.\n\n"
    "VALID LABELS (use exactly as written):\n"
    "  title               — the document's main title\n"
    "  section.title       — a section or sub-section heading\n"
    "  section.description — funder-provided context or instructions\n"
    "  question.text       — a specific data management requirement item\n"
    "  answer.text         — researcher's response or body text\n\n"
    "TEXT BLOCKS:\n"
    "{block_lines}\n\n"
    "OUTPUT: Reply with ONLY a JSON array — one object per block with "
    '"id" and "label". No other text.\n'
    "Example format: "
    '[{{"id": 0, "label": "title"}}, {{"id": 1, "label": "answer.text"}}]\n\n'
    "Your answer:"
)


def _file_to_b64(path: Path) -> str:
    """Read a PNG file from disk and return its base64 encoding."""
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _build_prompt(page_num: int, payload: list[dict]) -> str:
    return _PAGE_PROMPT_TEMPLATE.format(
        page_num=page_num,
        n=len(payload),
        payload=json.dumps(payload, ensure_ascii=False),
    )


def _build_ollama_prompt(page_num: int, payload: list[dict]) -> str:
    lines = "\n".join(
        f'  id={b["id"]}  bold={b["bold"]}  "{b["text"][:120]}"'
        for b in payload
    )
    return _OLLAMA_PROMPT_TEMPLATE.format(
        page_num=page_num,
        block_lines=lines,
    )


# ── Anthropic path ────────────────────────────────────────────────────────────

def _classify_page_anthropic(
    client,
    model: str,
    page_num: int,
    page_b64: str,
    payload: list[dict],
    max_tokens: int,
) -> list[dict]:
    """Send one page to Claude (Anthropic API) and return the parsed labels."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{
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
                {"type": "text", "text": _build_prompt(page_num, payload)},
            ],
        }],
    )
    raw = response.content[0].text if response.content else ""
    logger.info(
        "  page %d — in=%s  out=%s",
        page_num,
        f"{response.usage.input_tokens:,}",
        f"{response.usage.output_tokens:,}",
    )
    return parse_llm_json(raw, label=f"page={page_num}")


# ── Ollama path ───────────────────────────────────────────────────────────────

def _resize_b64_image(b64: str, max_px: int = 900) -> str:
    """Resize a base64 PNG so its longest side is at most *max_px*.

    Keeps the original file untouched; returns a new base64 string.
    Smaller images are returned unchanged.
    """
    import io
    from PIL import Image

    data = base64.standard_b64decode(b64)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _classify_page_ollama(
    host: str,
    model: str,
    page_num: int,
    page_b64: str,
    payload: list[dict],
) -> list[dict]:
    """Send one page to an Ollama vision model and return the parsed labels."""
    # Resize to ≤900 px — reduces visual tokens and avoids OOM on 7B models.
    # The original 150 DPI file on disk stays untouched for Claude reuse.
    small_b64 = _resize_b64_image(page_b64, max_px=900)

    resp = _requests.post(
        f"{host}/api/generate",
        json={
            "model":   model,
            "system":  SYSTEM_PROMPT,
            "prompt":  _build_ollama_prompt(page_num, payload),
            "images":  [small_b64],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 4096, "num_ctx": 8192},
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    raw  = data.get("response", "")
    logger.info(
        "  page %d — in=%s  out=%s",
        page_num,
        data.get("prompt_eval_count", "?"),
        data.get("eval_count", "?"),
    )
    return parse_llm_json(raw, label=f"page={page_num}")


# ── Strategy class ────────────────────────────────────────────────────────────

class VisionBatchStrategy:
    """Extract blocks with pdfplumber, classify page-by-page with image context.

    Each page is rendered as a PNG (saved to disk for reuse) and sent to the
    model alongside the text blocks extracted from that page.

    Parameters
    ----------
    model:
        Vision-capable model identifier.
        Anthropic: ``"claude-opus-4-8"``.
        Ollama: ``"qwen2-vl:7b"``, ``"llama3.2-vision:11b"``, etc.
    provider:
        ``"anthropic"`` or ``"ollama"``.  Defaults to ``"anthropic"``.
    host:
        Ollama base URL.  Only used when ``provider="ollama"``.
        Defaults to ``"http://localhost:11434"``.
    api_key:
        Anthropic API key.  Only used when ``provider="anthropic"``.
        Falls back to ``config.ANTHROPIC_API_KEY``.
    images_dir:
        Root directory where page PNGs are saved as
        ``{images_dir}/{pdf_stem}/page_001.png``.
        Pages already on disk are reused across experiments.
    resolution:
        DPI for page image rendering (default: 150).
    max_tokens:
        Max response tokens per page call — Anthropic only (default: 4096).
    """

    def __init__(
        self,
        model:      str = "claude-opus-4-8",
        provider:   str = "anthropic",
        host:       str = "http://localhost:11434",
        api_key:    str | None = None,
        images_dir: str | Path = "data/output/page_images",
        resolution: int = 150,
        max_tokens: int = 4096,
    ) -> None:
        if provider not in ("anthropic", "ollama"):
            raise ConfigurationError(
                f"vision_batch only supports provider='anthropic' or 'ollama', got {provider!r}."
            )

        self.model      = model
        self.provider   = provider
        self.images_dir = Path(images_dir)
        self.resolution = resolution
        self.max_tokens = max_tokens

        if provider == "anthropic":
            _key = api_key or config.ANTHROPIC_API_KEY
            if not _key:
                raise ConfigurationError(
                    "VisionBatchStrategy with provider='anthropic' requires ANTHROPIC_API_KEY."
                )
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed — run: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=_key)
            self._host   = None
        else:
            self._client = None
            self._host   = host.rstrip("/")
            # Verify Ollama is reachable and the model is available
            try:
                _requests.get(f"{self._host}/api/tags", timeout=5).raise_for_status()
            except _requests.exceptions.RequestException as exc:
                raise ConfigurationError(
                    f"Ollama not reachable at {self._host}.\n"
                    f"Start Ollama, then run:  ollama pull {model}\n"
                    f"Details: {exc}"
                ) from exc

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all blocks in *pdf_path* using page images."""
        logger.info("[vision_batch] extracting from %s …", pdf_path.name)
        blocks = extract_blocks(pdf_path)
        if not blocks:
            logger.warning("[vision_batch] no blocks found in %s", pdf_path.name)
            return []

        result = [dict(b) for b in blocks]
        pages  = sorted(set(b["page"] for b in result))

        img_dir   = self.images_dir / pdf_path.stem
        page_imgs = render_pages(pdf_path, img_dir, resolution=self.resolution)

        logger.info(
            "[vision_batch] %d blocks across %d pages — model=%s  provider=%s  images=%s",
            len(result), len(pages), self.model, self.provider, img_dir,
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

            if self.provider == "anthropic":
                parsed = _classify_page_anthropic(
                    self._client, self.model, page_num,
                    img_b64, payload, self.max_tokens,
                )
            else:
                parsed = _classify_page_ollama(
                    self._host, self.model, page_num,
                    img_b64, payload,
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
