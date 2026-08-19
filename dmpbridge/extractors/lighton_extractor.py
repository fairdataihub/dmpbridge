"""LightOnOCR-2-1B extractor — vision-LLM OCR pipeline.

Each PDF page is rendered as an image (via PyMuPDF), then passed through
the LightOnOCR-2-1B model (HuggingFace transformers) to produce OCR text.
The text is segmented line-by-line into block dicts.

OCR output carries no bbox metadata, so spatial fields are None for all
blocks. The model transcribes into markdown, so heading markers stand in for
font emphasis and set is_bold.

Install:
    pip install dmpbridge[lighton]
    # or directly:
    pip install transformers torch pymupdf
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import BaseExtractor

_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<text>.*)$")

DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"
_RENDER_DPI = 150   # 150 DPI keeps images small enough for 1B model inference

# The text backbone is Qwen3, which overflows in float16 and emits "!!!!" —
# bfloat16 is required for usable output. Resolved against torch at load time so
# this module stays importable without torch installed.
_MODEL_DTYPE_NAME = "bfloat16"

# Checkpoint submodule names -> the names transformers' Mistral3 implementation
# expects. Without this the vision half of the model loads as random weights.
_KEY_RENAMES = (
    ("model.vision_encoder.",    "model.vision_tower."),
    ("model.vision_projection.", "model.multi_modal_projector."),
)


def _remap_key(key: str) -> str:
    """Rename a checkpoint weight key to its transformers equivalent."""
    for old, new in _KEY_RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


class LightOnExtractor(BaseExtractor):
    """Run LightOnOCR-2-1B on every PDF page and segment output into blocks.

    Parameters
    ----------
    model_id:
        HuggingFace model repository, e.g. ``"lightonai/LightOnOCR-2-1B"``.
    max_new_tokens:
        Token budget for the model's OCR output per page.
    """

    def __init__(
        self,
        model_id:       str = DEFAULT_MODEL_ID,
        device:         str = "auto",
        max_new_tokens: int = 2048,
    ) -> None:
        self._max_new_tokens = max_new_tokens
        self._device         = self._resolve_device(device)
        self._processor, self._model = self._load_model(model_id)

    # ── BaseExtractor protocol ────────────────────────────────────────────────

    def extract(self, pdf_path: Path) -> list[dict]:
        pages  = self._render_pages(pdf_path)
        blocks: list[dict] = []
        for page_num, image in enumerate(pages, start=1):
            text = self._ocr_image(image)
            blocks.extend(self._text_to_blocks(text, page_num, len(blocks)))
        return blocks

    # ── Internal — setup ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. LightOnOCR requires a GPU.\n"
                "Check that your drivers and CUDA toolkit are installed."
            )
        return "cuda"

    @staticmethod
    def _load_model(model_id: str):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers / torch are not installed.\n"
                "Install with:  pip install dmpbridge[lighton]\n"
                "          or:  pip install transformers torch"
            ) from exc
        try:
            import fitz  # noqa: F401  — verify pymupdf is present early
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is not installed.\n"
                "Install with:  pip install pymupdf"
            ) from exc
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

        _log = logging.getLogger(__name__)
        _log.info("LightOnOCR : loading %s …", model_id)

        processor = AutoProcessor.from_pretrained(model_id)

        # The published checkpoint declares LightOnOCRForConditionalGeneration but
        # its config resolves to Mistral3, whose weight names differ. Loading via
        # from_pretrained silently randomises the vision tower (the model then
        # "sees" blank pages and reports no text), so build from config and load a
        # key-remapped state dict instead.
        dtype  = getattr(torch, _MODEL_DTYPE_NAME)
        config = AutoConfig.from_pretrained(model_id)
        model  = AutoModelForImageTextToText.from_config(config).to(dtype)

        state = load_file(hf_hub_download(model_id, "model.safetensors"))
        state = {_remap_key(k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.tie_weights()   # lm_head is tied to embed_tokens, never stored separately

        unresolved = [k for k in missing if k != "lm_head.weight"]
        if unresolved or unexpected:
            raise RuntimeError(
                f"LightOnOCR checkpoint did not load cleanly — "
                f"{len(unresolved)} missing, {len(unexpected)} unexpected weight(s). "
                f"First missing: {unresolved[:3]}; first unexpected: {list(unexpected)[:3]}. "
                f"The transformers version may have renamed the Mistral3 submodules again."
            )

        model = model.to("cuda").eval()
        used  = torch.cuda.memory_allocated() / 1024**3
        _log.info("LightOnOCR : loaded in %s — %.2f GB VRAM", _MODEL_DTYPE_NAME, used)

        return processor, model

    # ── Internal — inference ─────────────────────────────────────────────────

    @staticmethod
    def _render_pages(pdf_path: Path):
        """Render every page to a PIL Image at _RENDER_DPI."""
        import fitz
        from PIL import Image

        doc    = fitz.open(str(pdf_path))
        zoom   = _RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        return images

    def _ocr_image(self, image) -> str:
        import torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe all text in this document page exactly as it appears."},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )
        # Slice off the prompt tokens — keep only generated text
        generated = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self._processor.decode(generated[0], skip_special_tokens=True)

    @staticmethod
    def _text_to_blocks(text: str, page_num: int, offset: int) -> list[dict]:
        blocks: list[dict] = []
        for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
            # The model transcribes into markdown, so a heading marker is the only
            # emphasis signal available — treat it the way the other extractors
            # treat a bold font run, then drop the marker from the text itself.
            heading = _MD_HEADING_RE.match(line)
            if heading:
                line = heading.group("text").strip()
                if not line:
                    continue
            blocks.append({
                "page":          page_num,
                "line_order":    offset + len(blocks),
                "text":          line,
                "x0":            None,
                "top":           None,
                "x1":            None,
                "bottom":        None,
                "avg_font_size": None,
                "font_names":    [],
                "is_bold":       bool(heading),
                "is_italic":     False,
            })
        return blocks
