"""LLM-based classifier using Ollama."""

import json
import logging

import requests

from . import config

logger = logging.getLogger(__name__)

LABELS = ("document_title", "section", "subsection", "content")

# JSON schema that constrains Ollama to output a well-formed array (Ollama ≥ 0.5)
_OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":    {"type": "integer"},
            "label": {"type": "string", "enum": list(LABELS)},
        },
        "required": ["id", "label"],
    },
}

SYSTEM_PROMPT = """You are a document structure classifier.

Label each text block with exactly one of:
- document_title : ONLY the single main title of the entire document. Appears ONCE on page 1. If unsure, use "section" instead.
- section        : Major heading (e.g. "1. Introduction", "CHAPTER 2", "Abstract", "References", "Conclusion")
- subsection     : Sub-heading within a section (e.g. "1.1 Background", "2.3.1 Results", "A. Notation")
- content        : Everything else — body text, paragraphs, captions, footnotes, tables, figures, page numbers

Rules:
- Use "document_title" at most once per batch and ONLY if the text is clearly the document's main title on page 1
- Larger font + bold + short text → heading (section or subsection)
- Long sentences or multiple lines → content
- When in doubt between document_title and section, choose section

You MUST output a JSON array with one entry for EVERY block — no explanation, no markdown.
Example: [{"id": 0, "label": "section"}, {"id": 1, "label": "content"}, {"id": 2, "label": "subsection"}]
"""

BATCH_SIZE = config.BATCH_SIZE  # set in config.py


class OllamaClassifier:
    """Classifies document blocks using a locally running Ollama model."""

    def __init__(self, model: str = config.MODEL, host: str = config.HOST):
        self.model = model
        self.host = host.rstrip("/")
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as exc:
            raise ConnectionError(
                f"Ollama is not reachable at {self.host}.\n"
                "Install and start it: https://ollama.com\n"
                f"Then pull the model:  ollama pull {self.model}\n"
                f"Details: {exc}"
            ) from exc

    def classify_blocks(self, blocks: list[dict]) -> list[dict]:
        """Return blocks with the 'label' field populated."""
        result = [dict(b) for b in blocks]

        for start in range(0, len(result), BATCH_SIZE):
            batch = result[start : start + BATCH_SIZE]
            logger.info(f"  Classifying blocks {start}–{start + len(batch) - 1} …")
            labels = self._classify_batch(batch, offset=start)
            for entry in labels:
                idx = entry.get("id")
                lbl = entry.get("label", "content")
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"] = lbl

        return result

    def _classify_batch(self, batch: list[dict], offset: int) -> list[dict]:
        payload = [
            {
                "id": offset + j,
                "text": b["text"],
                "size": round(b["avg_font_size"], 1),
                "bold": b["is_bold"],
                "page": b["page"],
            }
            for j, b in enumerate(batch)
        ]

        prompt = (
            f"Classify ALL {len(batch)} blocks below and return a JSON array "
            f"with exactly {len(batch)} entries, one per block:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "format": _OUTPUT_SCHEMA,
                "options": {"temperature": 0.0},
            },
            timeout=300,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        parsed = json.loads(raw) if raw else []

        if not isinstance(parsed, list) or not parsed:
            logger.warning(f"Empty response from LLM for batch at offset {offset}.")
            return []

        return parsed
