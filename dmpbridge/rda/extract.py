"""One model call: a plan's text -> RDA maDMP 1.2 JSON.

The prompt is ``prompt.txt``, in the format the RDA notebooks save with every run::

    SYSTEM:
    <the system prompt>

    PROMPT:
    <the user prompt, with {dmp_text} where the plan's text goes>

A ``<<TODAY>>`` placeholder in the system prompt is replaced with the current date
when the prompt is sent, so a fallback date is the day the pipeline runs rather than
a date written into the file. The complete, unmodified maDMP 1.2 schema
(``maDMP-schema-1.2.json``, ``$ref`` pointers resolved) is not in the prompt - it is
Ollama's structured-output ``format``, which constrains the answer to the schema's
structure. Versions up to v10b also pasted the schema into the prompt as text; v10
dropped it, cutting the prompt from about 11,700 tokens to about 2,400. The versions
are in scripts/build/prompts/ and their results in notebooks/rda-experiments/runs/.

The settings match the notebook runs the prompt was evaluated with:
temperature 0, a 32k context, an 8000-token answer cap, and a repeat-penalty window
of 1024 tokens (with Ollama's default 64, llama3.1:8b repeated one dataset block
until the token cap).
"""
import hashlib
import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from ..core import config
from ..models.ollama import OllamaModel

HERE = Path(__file__).parent
PROMPT_VERSION = "v11"                      # see notebooks/rda-version-comparison.ipynb for how the versions score
PROMPT_FILE = HERE / "prompt.txt"
SCHEMA_FILE = HERE / "maDMP-schema-1.2.json"

NUM_CTX = 32768
MAX_TOKENS = 8000
OPTIONS = {"repeat_last_n": 1024}


class RdaExtractionError(Exception):
    """The model's answer was not valid JSON. ``raw`` holds what it wrote."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


@lru_cache(maxsize=1)
def load_schema() -> tuple[dict, str]:
    """The maDMP 1.2 schema with ``$ref`` resolved, as an object and as compact text."""
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    full = resolve(schema)
    return full, json.dumps(full, separators=(",", ":"))


_PROMPT_HEADER = re.compile(r"^PROMPT(?: \([^)\n]*\))?:[ \t]*$", re.M)   # "PROMPT:" or "PROMPT (…):"


def parse_prompt_file(text: str) -> tuple[str, str]:
    """Split a prompt file into ``(system prompt, user template)``.

    The system part is everything between the ``SYSTEM:`` line and the ``PROMPT:``
    line, kept exactly (the prompts are whitespace-sensitive) except for the blank
    lines that separate the two parts. The user template must contain ``{dmp_text}``.
    """
    if text.startswith("SYSTEM:\n"):
        text = text[len("SYSTEM:\n"):]
    m = _PROMPT_HEADER.search(text)
    if m is None:
        raise ValueError(f"{PROMPT_FILE}: no 'PROMPT:' line separating the system and user prompts")
    system = text[:m.start()].rstrip("\n") + "\n"
    user = text[m.end():].strip("\n")
    if "{dmp_text}" not in user:
        raise ValueError(f"{PROMPT_FILE}: the user prompt has no {{dmp_text}} placeholder")
    return system, user


@lru_cache(maxsize=1)
def _prompt_parts() -> tuple[str, str]:
    return parse_prompt_file(PROMPT_FILE.read_text(encoding="utf-8"))


def system_prompt() -> str:
    """The system prompt as sent: the file's text with ``<<TODAY>>`` replaced by the
    current date. Otherwise exactly as stored - never stripped or reformatted."""
    return _prompt_parts()[0].replace("<<TODAY>>", date.today().isoformat())


def make_prompt(plan_text: str) -> str:
    """The user prompt: the file's template with the plan's text in place of ``{dmp_text}``."""
    return _prompt_parts()[1].replace("{dmp_text}", plan_text)


def prompt_fingerprint() -> str:
    """A short hash of the prompt file's text (before the date goes in). A version name
    alone cannot tell two edits of the same version apart; this can, so a saved stage-5
    file made with an earlier wording is re-run rather than reused - while the date
    changing from one day to the next does not force a re-run."""
    system, user = _prompt_parts()
    return hashlib.sha256((system + user).encode("utf-8")).hexdigest()[:12]


def extract_rda(plan_text: str, model: str, host: str = config.HOST) -> tuple[dict, dict]:
    """Extract the RDA maDMP JSON from a plan's text with *model*.

    Returns ``(rda_json, call_info)``; ``call_info`` holds Ollama's token counts
    and timings. Raises :class:`RdaExtractionError` when the answer is not valid
    JSON - usually a model that looped until the token cap.
    """
    schema, _ = load_schema()
    llm = OllamaModel(model=model, host=host, num_ctx=NUM_CTX, num_predict=MAX_TOKENS, options=OPTIONS)
    raw = llm.complete(system_prompt(), make_prompt(plan_text), schema=schema)
    info = dict(llm.last_call or {})
    try:
        return json.loads(raw), info
    except json.JSONDecodeError as e:
        why = ("stopped at the token cap" if info.get("done_reason") == "length"
               else f"{e.msg} at character {e.pos}")
        raise RdaExtractionError(f"{model} did not return valid JSON ({why})", raw) from None


def merge(narrative: dict, rda: dict) -> dict:
    """The combined document: the RDA JSON's top level (``dmp``) first, then the
    narrative JSON's top level (``narrative``), each unchanged."""
    overlap = set(narrative) & set(rda)
    if overlap:
        raise ValueError(f"narrative and RDA JSON share top-level keys: {sorted(overlap)}")
    return {**rda, **narrative}
