"""LightOnOCR-2-1B extractor — vision-LLM OCR pipeline.

Each PDF page is rendered as an image (via PyMuPDF), then passed through
the LightOnOCR-2-1B model (HuggingFace transformers) to produce OCR text.
The text is segmented line-by-line into block dicts.

Because OCR output carries no font or bbox metadata, is_bold is always
False and spatial fields are None for all blocks.

Install:
    pip install dmpbridge[lighton]
    # or directly:
    pip install transformers torch pymupdf
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseExtractor

DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"
_RENDER_DPI = 150   # 150 DPI keeps images small enough for 1B model inference


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
            from transformers import AutoModelForImageTextToText, AutoProcessor
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

        import logging
        logging.getLogger("transformers").setLevel(logging.WARNING)

        _log = logging.getLogger(__name__)
        n_gpus = torch.cuda.device_count()
        _log.info("LightOnOCR : %d GPU(s) detected — loading %s with device_map=auto", n_gpus, model_id)

        processor = AutoProcessor.from_pretrained(model_id)
        # device_map="auto" spreads model layers across all available GPUs
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        for i in range(n_gpus):
            used  = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            _log.info("LightOnOCR : GPU %d — %.2f / %.1f GB VRAM used", i, used, total)

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
        return [
            {
                "page":          page_num,
                "line_order":    offset + i,
                "text":          line,
                "x0":            None,
                "top":           None,
                "x1":            None,
                "bottom":        None,
                "avg_font_size": None,
                "font_names":    [],
                "is_bold":       False,
                "is_italic":     False,
                "label":         None,
            }
            for i, line in enumerate(
                ln.strip() for ln in text.splitlines() if ln.strip()
            )
        ]
