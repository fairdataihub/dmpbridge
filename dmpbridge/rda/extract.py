"""One model call: a plan's text -> RDA maDMP 1.2 JSON.

The prompt is version v10c, developed and evaluated in
notebooks/pdf-to-rda-dmp-json-narrative.ipynb: the v10b instructions as the system
prompt (``prompt_v10b.txt``, a byte-for-byte copy of
scripts/build/prompts/rda_narrative_v10b.txt, which is where earlier versions live),
and a user prompt that holds only the plan's text. The complete, unmodified maDMP 1.2
schema (``maDMP-schema-1.2.json``, ``$ref`` pointers resolved) is not in the prompt -
it is Ollama's structured-output ``format``, which constrains the answer to the
schema's structure. Earlier versions also pasted the schema into the prompt as text;
v10c dropped it, cutting the prompt from about 11,700 tokens to about 2,400.

The settings match the notebook runs the prompt was evaluated with:
temperature 0, a 32k context, an 8000-token answer cap, and a repeat-penalty window
of 1024 tokens (with Ollama's default 64, llama3.1:8b repeated one dataset block
until the token cap).
"""
import json
from functools import lru_cache
from pathlib import Path

from ..core import config
from ..models.ollama import OllamaModel

HERE = Path(__file__).parent
PROMPT_VERSION = "v10c"
PROMPT_FILE = HERE / "prompt_v10b.txt"      # v10c uses v10b's instructions unchanged
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


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """The RDA system prompt (:data:`PROMPT_VERSION`), exactly as stored - it is
    whitespace-sensitive, so it is never stripped or reformatted."""
    return PROMPT_FILE.read_text(encoding="utf-8")


def make_prompt(plan_text: str) -> str:
    """The user prompt: the plan's text only - the same wording as the notebook's v10c run."""
    return f"""Here is the text of a Data Management Plan:

{plan_text}

Generate one JSON object that follows the JSON form in your instructions, filling in
whatever information the Data Management Plan text contains. Output only the JSON."""


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
