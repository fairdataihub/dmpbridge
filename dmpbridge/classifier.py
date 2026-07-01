"""LLM-based classifier — supports Ollama, OpenAI, Anthropic, and Google Gemini."""

# What this file does — step by step:
#   Step 1 — define the 5 allowed labels and a JSON schema that forces structured output
#   Step 2 — build the system prompt: label definitions, decision rules, and real examples
#   Step 3 — split all blocks into small batches so each request fits the model's context window
#   Step 4 — for each batch, attach the last 3 already-labeled blocks as context
#   Step 5 — send system prompt + context + batch to the selected LLM at temperature 0
#   Step 6 — parse the returned [{id, label}] array and write each label back into the block list
#   Step 7 — return the complete block list with every label filled in

import json
import logging

import requests

from . import config


logger = logging.getLogger(__name__)

# The 5 labels the LLM is allowed to assign to each text block.
# Every block in a DMP must get exactly one of these.
LABELS = ("title", "section.title", "section.description", "question.text", "answer.text")

# JSON schema used by Ollama to enforce the output format.
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

# The system prompt sent to the LLM before any document text.
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

# How many blocks to send to the LLM in one request.
# Larger batches are faster but risk hitting the model's context window limit.
BATCH_SIZE = config.BATCH_SIZE

# How many already-labeled blocks to include as context before each new batch.
# This helps the model maintain continuity — e.g., if the last labeled block
# was a section.title, the model can correctly infer what comes next.
CONTEXT_SIZE = 3


# ── Base class ────────────────────────────────────────────────────────────────

class BaseClassifier:
    """Shared batching and context-window logic. Subclasses only implement _classify_batch."""

    def classify_blocks(self, blocks: list[dict]) -> list[dict]:
        """Classify all blocks in document order, in batches, with sliding context. 1. Copy the original blocks. 2. Process in batches of BATCH_SIZE. 3. For each batch, pass the last 3 labeled blocks as context. 4. Update the result with labels returned by the provider."""
        # Instead of changing the original blocks, make a copy.
        result = [dict(b) for b in blocks]

        for start in range(0, len(result), BATCH_SIZE):
            batch   = result[start : start + BATCH_SIZE]
            # Grab the last few already-labeled blocks so the model knows where it is in the document.
            context = result[max(0, start - CONTEXT_SIZE) : start]

            logger.info(f"  Classifying blocks {start}–{start + len(batch) - 1} …")
            labels = self._classify_batch(batch, offset=start, context=context)

            # Write each returned label back into the result list.
            for entry in labels:
                idx = entry.get("id")
                lbl = entry.get("label", "answer.text")
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"] = lbl

        return result

    def _classify_batch(self, batch: list[dict], offset: int, context: list[dict] | None = None) -> list[dict]:
        raise NotImplementedError

    def _build_user_prompt(self, batch: list[dict], offset: int, context: list[dict] | None) -> str:
        """Build the user-facing prompt: optional labeled context + blocks to classify."""
        ctx_section = ""
        if context:
            ctx_blocks = [
                {
                    "text":   b["text"],
                    "bold":   b["is_bold"],
                    "italic": b.get("is_italic", False),
                    "label":  b.get("label", "?"),
                }
                for b in context
            ]
            ctx_section = (
                "PRECEDING BLOCKS (already labeled — use for context only, do not output labels for these):\n"
                + json.dumps(ctx_blocks, ensure_ascii=False)
                + "\n\n"
            )

        payload = [
            {
                "id":     offset + j,
                "text":   b["text"],
                "bold":   b["is_bold"],
                "italic": b.get("is_italic", False),
                "page":   b["page"],
            }
            for j, b in enumerate(batch)
        ]

        return (
            ctx_section
            + f"TO CLASSIFY — return a JSON array with exactly {len(batch)} entries, one per block:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    def _parse_json(self, raw: str, offset: int) -> list[dict]:
        """Parse the LLM's JSON response. Handles bare arrays and wrapped objects like {"labels": [...]}."""
        if not raw:
            logger.warning(f"Empty response from LLM for batch at offset {offset}.")
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response for batch at offset {offset}.")
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Unwrap {"labels": [...]} or any dict whose first list value is the result.
            for v in data.values():
                if isinstance(v, list):
                    return v
        logger.warning(f"Unexpected response shape for batch at offset {offset}.")
        return []


# ── Provider implementations ──────────────────────────────────────────────────

class OllamaClassifier(BaseClassifier):
    """Classifies blocks using a locally running Ollama model."""

    def __init__(self, model: str = config.MODEL, host: str = config.HOST):
        self.model = model
        self.host  = host.rstrip("/")
        self._verify_connection()

    def _verify_connection(self) -> None:
        # Quick check — fails early with a clear message if Ollama is not running.
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

    def _classify_batch(self, batch: list[dict], offset: int, context: list[dict] | None = None) -> list[dict]:
        """Send one batch to Ollama with JSON schema enforcement and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
        # Send to Ollama — format=JSON schema forces it to return the exact shape we need.
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model":   self.model,
                "system":  SYSTEM_PROMPT,
                "prompt":  prompt,
                "stream":  False,
                "format":  _OUTPUT_SCHEMA,
                "options": {"temperature": 0.0},
            },
            timeout=300,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return self._parse_json(raw, offset)


class OpenAIClassifier(BaseClassifier):
    """Classifies blocks using the OpenAI API (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "The openai package is not installed.\n"
                "Install it with:  pip install openai"
            )
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set.\n"
                "Add it to your .env file:  OPENAI_API_KEY=sk-..."
            )
        self.client = _openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model  = model

    def _classify_batch(self, batch: list[dict], offset: int, context: list[dict] | None = None) -> list[dict]:
        """Send one batch to OpenAI with json_object mode and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
        # json_object mode requires a dict output — ask for {"labels": [...]} wrapper so it's reliable.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + '\n\nWrap your array in a JSON object: {"labels": [...]}',
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        return self._parse_json(raw, offset)


class AnthropicClassifier(BaseClassifier):
    """Classifies blocks using the Anthropic API (Claude models)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is not installed.\n"
                "Install it with:  pip install anthropic"
            )
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Add it to your .env file:  ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model  = model

    def _classify_batch(self, batch: list[dict], offset: int, context: list[dict] | None = None) -> list[dict]:
        """Send one batch to Anthropic and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.content[0].text if response.content else ""
        return self._parse_json(raw, offset)


class GeminiClassifier(BaseClassifier):
    """Classifies blocks using the Google Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        try:
            import google.generativeai as _genai
        except ImportError:
            raise ImportError(
                "The google-generativeai package is not installed.\n"
                "Install it with:  pip install google-generativeai"
            )
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set.\n"
                "Add it to your .env file:  GEMINI_API_KEY=AIza..."
            )
        _genai.configure(api_key=config.GEMINI_API_KEY)
        self._genai     = _genai
        self.model_name = model

    def _classify_batch(self, batch: list[dict], offset: int, context: list[dict] | None = None) -> list[dict]:
        """Send one batch to Gemini with JSON mime type enforcement and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
        # Build a fresh model instance per batch — Gemini's GenerativeModel is lightweight.
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=self._genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        response = model.generate_content(prompt)
        raw = response.text or ""
        return self._parse_json(raw, offset)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_classifier(
    provider: str | None = None,
    model:    str | None = None,
    host:     str | None = None,
) -> BaseClassifier:
    """Return the right classifier for the given provider. Falls back to config defaults if not specified."""
    provider = (provider or config.PROVIDER).lower()
    model    = model or config.MODEL

    if provider == "ollama":
        return OllamaClassifier(model=model, host=host or config.HOST)
    if provider == "openai":
        return OpenAIClassifier(model=model)
    if provider == "anthropic":
        return AnthropicClassifier(model=model)
    if provider == "gemini":
        return GeminiClassifier(model=model)

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        "Choose from: ollama, openai, anthropic, gemini"
    )
