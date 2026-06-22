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

# ── Improvement 1: few-shot examples drawn from manually labeled DMPs ────────
_FEW_SHOT_EXAMPLES = """
EXAMPLES FROM REAL DMP DOCUMENTS:

title:
  "Center for Bio-Inspired Energy Science"
  "CAREER: HIGH-RESOLUTION NMR FOR PARAMAGNETIC SODIUM ELECTRODES"

section.title:
  "1. Data sharing and preservation"
  "Element 1: Data Type:"
  "2. Data used in publications"
  "Products of Research"

section.description (funder template text — instructs the author what to write):
  "Data management plans should describe whether and how data generated in the course of the proposed research will be shared and preserved."
  "Data management plans should provide a plan for making all research data displayed in publications resulting from the proposed research open, machine-readable, and digitally accessible to the public at the time of publication."
  "Data management plans must protect confidentiality, personal privacy, Personally Identifiable Information and U.S. national, homeland, and economic security."

question.text (sub-question prompt inside a section — asks the author to address a specific topic):
  "A. Types and amount of scientific data expected to be generated in the project:"
  "A. Repository where scientific data and metadata will be archived:"
  "B. Whether access to scientific data will be controlled:"
  "Roles & Responsibilities. For the proposed research, describe who will be responsible for coordinating and ensuring data storage and access."
  "Data Types and Sources. A brief, high-level description of the data to be generated."

answer.text (researcher's actual written response — narrative, first-person, describes what the team will do):
  "This secondary data analysis project will analyze deidentified data from 48,218 participants from eight studies and the publicly available NHANES cohorts."
  "All data pertaining to any published work will be made available to any person on request. Numerical published data as well as the original raw data files will be freely accessible."
  "Data will be analyzed with custom code by our statistical and computer science team."
"""

SYSTEM_PROMPT = f"""You are a classifier for Data Management Plan (DMP) documents.

Label each text block with exactly one of these 5 labels:

- title              : The single main title of the entire document. Appears once, very short.
- section.title      : A numbered or named section heading (e.g. "1. Data sharing", "Element 1: Data Type:", "Products of Research").
- section.description: Funder template text that instructs the author what to write in this section. Uses words like "should", "must", "DMPs should", "provide a plan for". This is NOT written by the researcher.
- question.text      : A sub-question or sub-topic prompt inside a section. Asks the author to address a specific topic. Often starts with a letter prefix (A., B.) or a bold phrase. Always appears after a section.title.
- answer.text        : The researcher's actual written response — narrative paragraphs describing what the team will do, has done, or plans to do.

Key distinctions:
- section.description = funder wrote it (requirements/instructions); answer.text = researcher wrote it (actual plans/responses)
- question.text = specific sub-topic prompt inside a section; section.description = overall section requirements
- If a block uses "should" or "must" and sounds like instructions → section.description
- If a block describes what the research team will actually do → answer.text
{_FEW_SHOT_EXAMPLES}
You MUST output a JSON array with one entry for EVERY block in the TO CLASSIFY list — no explanation, no markdown.
"""

BATCH_SIZE = config.BATCH_SIZE
CONTEXT_SIZE = 3  # number of preceding labeled blocks sent as context


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
            # Improvement 2: pass last CONTEXT_SIZE labeled blocks as context
            context = result[max(0, start - CONTEXT_SIZE) : start]
            logger.info(f"  Classifying blocks {start}–{start + len(batch) - 1} …")
            labels = self._classify_batch(batch, offset=start, context=context)
            for entry in labels:
                idx = entry.get("id")
                lbl = entry.get("label", "answer.text")
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"] = lbl

        return result

    def _classify_batch(
        self, batch: list[dict], offset: int, context: list[dict] | None = None
    ) -> list[dict]:
        # Build context prefix (already-labeled blocks — read only)
        ctx_section = ""
        if context:
            ctx_blocks = [
                {
                    "text": b["text"],
                    "bold": b["is_bold"],
                    "italic": b.get("is_italic", False),
                    "label": b.get("label", "?"),
                }
                for b in context
            ]
            ctx_section = (
                "PRECEDING BLOCKS (already labeled — use for context only, do not output labels for these):\n"
                + json.dumps(ctx_blocks, ensure_ascii=False)
                + "\n\n"
            )

        # Blocks to classify
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
            ctx_section
            + f"TO CLASSIFY — return a JSON array with exactly {len(batch)} entries, one per block:\n"
            + json.dumps(payload, ensure_ascii=False)
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
