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

import requests

from . import config
from .exceptions import ConfigurationError, ProviderConnectionError
from .logging_setup import get_logger
from .prompt import LABELS, OUTPUT_SCHEMA, SYSTEM_PROMPT, build_batch_prompt

logger = get_logger(__name__)

# Re-exported for callers that imported from this module before prompt.py existed.
_OUTPUT_SCHEMA = OUTPUT_SCHEMA

# How many blocks to send to the LLM in one request.
# Larger batches are faster but risk hitting the model's context window limit.
BATCH_SIZE   = config.BATCH_SIZE
CONTEXT_SIZE = 3


# ── Base class ────────────────────────────────────────────────────────────────

class BaseClassifier:
    """Shared batching and context-window logic. Subclasses only implement _classify_batch."""

    def __init__(
        self,
        batch_size:   int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        self.batch_size   = batch_size
        self.context_size = context_size

    def classify_blocks(self, blocks: list[dict]) -> list[dict]:
        """Classify all blocks in document order, in batches, with sliding context.

        1. Copy the original blocks.
        2. Process in batches of self.batch_size.
        3. For each batch, pass the last context_size labeled blocks as context.
        4. Update the result with labels returned by the provider.
        """
        result = [dict(b) for b in blocks]

        for start in range(0, len(result), self.batch_size):
            batch   = result[start : start + self.batch_size]
            context = result[max(0, start - self.context_size) : start]

            logger.info("  Classifying blocks %d–%d …", start, start + len(batch) - 1)
            labels = self._classify_batch(batch, offset=start, context=context)

            for entry in labels:
                idx = entry.get("id")
                lbl = entry.get("label", "answer.text")
                if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
                    result[idx]["label"] = lbl

        return result

    def _classify_batch(
        self,
        batch: list[dict],
        offset: int,
        context: list[dict] | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    def _build_user_prompt(
        self,
        batch: list[dict],
        offset: int,
        context: list[dict] | None,
    ) -> str:
        """Build the user-facing prompt: optional labeled context + blocks to classify."""
        return build_batch_prompt(batch, offset, context)

    def _parse_json(self, raw: str, offset: int) -> list[dict]:
        """Parse the LLM's JSON response.

        Handles bare arrays and wrapped objects like {"labels": [...]}.
        """
        if not raw:
            logger.warning("Empty response from LLM for batch at offset %d.", offset)
            return []
        # Strip markdown code fences that some providers (e.g. Anthropic) wrap around JSON.
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```", 2)[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
            raw = stripped.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response for batch at offset %d.", offset)
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        logger.warning("Unexpected response shape for batch at offset %d.", offset)
        return []


# ── Provider implementations ──────────────────────────────────────────────────

class OllamaClassifier(BaseClassifier):
    """Classifies blocks using a locally running Ollama model."""

    def __init__(
        self,
        model:        str = config.MODEL,
        host:         str = config.HOST,
        batch_size:   int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        super().__init__(batch_size=batch_size, context_size=context_size)
        self.model = model
        self.host  = host.rstrip("/")
        self._verify_connection()

    def _verify_connection(self) -> None:
        # Quick check — fails early with a clear message if Ollama is not running.
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ProviderConnectionError(
                f"Ollama is not reachable at {self.host}.\n"
                "Install and start it: https://ollama.com\n"
                f"Then pull the model:  ollama pull {self.model}\n"
                f"Details: {exc}"
            ) from exc

    def _classify_batch(
        self,
        batch: list[dict],
        offset: int,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """Send one batch to Ollama with JSON schema enforcement and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
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

    def __init__(
        self,
        model:        str = "gpt-4o",
        batch_size:   int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        super().__init__(batch_size=batch_size, context_size=context_size)
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "The openai package is not installed.\n"
                "Install it with:  pip install openai"
            )
        if not config.OPENAI_API_KEY:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set.\n"
                "Add it to your .env file:  OPENAI_API_KEY=sk-..."
            )
        self.client = _openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model  = model

    def _classify_batch(
        self,
        batch: list[dict],
        offset: int,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """Send one batch to OpenAI with json_object mode and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
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

    def __init__(
        self,
        model:        str = "claude-sonnet-4-6",
        batch_size:   int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        super().__init__(batch_size=batch_size, context_size=context_size)
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is not installed.\n"
                "Install it with:  pip install anthropic"
            )
        if not config.ANTHROPIC_API_KEY:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Add it to your .env file:  ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model  = model

    def _classify_batch(
        self,
        batch: list[dict],
        offset: int,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """Send one batch to Anthropic and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else ""
        return self._parse_json(raw, offset)


class GeminiClassifier(BaseClassifier):
    """Classifies blocks using the Google Gemini API."""

    def __init__(
        self,
        model:        str = "gemini-2.0-flash",
        batch_size:   int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
    ) -> None:
        super().__init__(batch_size=batch_size, context_size=context_size)
        try:
            import google.generativeai as _genai
        except ImportError:
            raise ImportError(
                "The google-generativeai package is not installed.\n"
                "Install it with:  pip install google-generativeai"
            )
        if not config.GEMINI_API_KEY:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set.\n"
                "Add it to your .env file:  GEMINI_API_KEY=AIza..."
            )
        _genai.configure(api_key=config.GEMINI_API_KEY)
        self._genai     = _genai
        self.model_name = model

    def _classify_batch(
        self,
        batch: list[dict],
        offset: int,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """Send one batch to Gemini with JSON mime type enforcement and return the predicted labels."""
        prompt = self._build_user_prompt(batch, offset, context)
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
    provider:     str | None = None,
    model:        str | None = None,
    host:         str | None = None,
    batch_size:   int = BATCH_SIZE,
    context_size: int = CONTEXT_SIZE,
) -> BaseClassifier:
    """Return the right classifier for the given provider. Falls back to config defaults."""
    provider = (provider or config.PROVIDER).lower()
    model    = model or config.MODEL

    if provider == "ollama":
        return OllamaClassifier(model=model, host=host or config.HOST,
                                batch_size=batch_size, context_size=context_size)
    if provider == "openai":
        return OpenAIClassifier(model=model,
                                batch_size=batch_size, context_size=context_size)
    if provider == "anthropic":
        return AnthropicClassifier(model=model,
                                   batch_size=batch_size, context_size=context_size)
    if provider == "gemini":
        return GeminiClassifier(model=model,
                                batch_size=batch_size, context_size=context_size)

    raise ConfigurationError(
        "Unknown provider: %r. Choose from: ollama, openai, anthropic, gemini" % provider
    )
