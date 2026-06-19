"""LLM-based classifier using Ollama."""

import json
import logging

import requests

from . import config

logger = logging.getLogger(__name__)

LABELS = ("title", "section.title", "section.description", "question.text", "answer.text")

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

SYSTEM_PROMPT = """You are a classifier for Data Management Plan (DMP) documents.

Label each text block with exactly one of these 5 labels:

- title              : The single main title of the entire document. Appears once, very short, no section number.
- section.title      : A numbered section heading that names a major section of the DMP (e.g. "1. Data sharing and preservation", "Element 1: Data Type:", "Roles and responsibilities","REVIEW OF PROPOSAL COMPONENTS","TYPES OF DATA", "1. Types of data", "1. Policy and Practice", "Products of Research").
- section.description: Template or guideline text written by the funder that describes what the section must cover. This is instructional text directed at the author — it explains requirements and expectations. Often uses words like "should", "must", "DMPs should", "provide a plan for", "describe whether".
- question.text      : A sub-question or sub-topic prompt within a section. Introduces a specific topic the author must address. Often starts with a short bold phrase or title followed by an explanatory sentence (e.g. "A. Types and amount of scientific data expected to be generated in the project:","Roles & Responsibilities.","Data Types and Sources. A brief, high-level description of the data to be generated or used through the course of the proposed research and which of these are considered Digital Research Datanecessary to Validate the research findings.").
- answer.text        : The researcher's actual written response. This is the content authored by the DMP writer — narrative paragraphs, explanations, plans, and descriptions of what the research team will actually do.

Key distinctions:
- section.description is funder/template text (what must be written); answer.text is researcher text (what was written)
- question.text introduces a specific sub-topic; section.description describes the whole section's requirements
- title and section.title are very short; everything else is longer

You MUST output a JSON array with one entry for EVERY block — no explanation, no markdown.
Example: [{"id": 0, "label": "section.title"}, {"id": 1, "label": "section.description"}, {"id": 2, "label": "answer.text"}]
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
                lbl = entry.get("label", "answer.text")
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"] = lbl

        return result

    def _classify_batch(self, batch: list[dict], offset: int) -> list[dict]:
        payload = [
            {
                "id": offset + j,
                "text": b["text"],
                "bold": b["is_bold"],
                "italic": b.get("is_italic", False),
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
