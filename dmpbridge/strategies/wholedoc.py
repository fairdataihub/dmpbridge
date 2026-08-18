"""Whole-document strategy — configurable extraction + single model call.

All blocks are sent to the model in one request so it sees the full document
at once, improving structural continuity at the cost of higher token usage.

The extraction step is handled by a :class:`~dmpbridge.extractors.BaseExtractor`
so the pipeline can compare pdfplumber and LightOnOCR side-by-side without
changing any other code.

Example
-------
    from pathlib import Path
    from dmpbridge.strategies.wholedoc import WholeDocStrategy

    strategy = WholeDocStrategy(model="llama3.3:70b", extractor="lighton")
    blocks   = strategy.run(Path("document.pdf"))
"""
import json
import re
from pathlib import Path
from typing import Optional

from ..core import config
from ..extractors import get_extractor
from ..models import get_model
from ..parsers import parse_llm_json
from ..prompts import LABELS, SYSTEM_PROMPT, labels as prompt_labels
from ..utils import ConfigurationError, get_logger

logger = get_logger(__name__)


def _build_wholedoc_prompt(payload: list[dict]) -> str:
    return (
        f"CLASSIFY ALL BLOCKS — return a JSON array with exactly {len(payload)} entries, "
        f"one per block, in the same order:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def clean_labeled_output(labeled_data, verbose=True):
    """Remove empty entries and normalize whitespace (strip \\n, \\t, extra spaces).

    Name, signature and logic copied verbatim from the notebook's
    clean_labeled_output() — including that it does not check whether
    ``label`` is one of the five known values, only that ``text`` is
    non-empty after normalizing.
    """
    cleaned = []
    for item in labeled_data:
        text = item.get("text", "")
        normalized = re.sub(r"\s+", " ", text).strip()

        if normalized:
            cleaned.append({"text": normalized, "label": item.get("label")})
        elif verbose:
            print(
                f"Dropped empty entry with label: {item.get('label')}"
            )  # just to check what it drops

    return cleaned


def _build_payload(blocks: list[dict]) -> list[dict]:
    return [
        {
            "id":     j,
            "text":   b["text"],
            "bold":   b["is_bold"],
            "italic": b.get("is_italic", False),
            "page":   b["page"],
        }
        for j, b in enumerate(blocks)
    ]


def _apply_labels(blocks: list[dict], parsed: list[dict]) -> list[dict]:
    result = [dict(b) for b in blocks]
    for entry in parsed:
        idx  = entry.get("id")
        lbl  = entry.get("label", "answer.text")
        conf = float(entry.get("confidence", 1.0))
        if idx is not None and 0 <= idx < len(result) and lbl in LABELS:
            result[idx]["label"]      = lbl
            result[idx]["confidence"] = max(0.0, min(1.0, conf))
    for b in result:
        if not b.get("label"):
            b["label"] = "answer.text"
        if "confidence" not in b:
            b["confidence"] = 1.0
    return result


class WholeDocStrategy:
    """Extract blocks from a PDF then classify them in a single model call.

    The extraction step is delegated to a :class:`~dmpbridge.extractors.BaseExtractor`
    so that pdfplumber and LightOnOCR can be swapped in without touching the
    prompt or evaluation code.

    Parameters
    ----------
    provider:
        LLM provider — only ``"ollama"`` is supported.
    model:
        Ollama model identifier (e.g. ``"llama3.3:70b"``).
    host:
        Ollama base URL.
    extractor:
        PDF extraction backend — ``"pdfplumber"`` (default) or ``"lighton"``.
    """

    def __init__(
        self,
        provider:   str = config.PROVIDER,
        model:      str = config.MODEL,
        host:       str = config.HOST,
        extractor:  str = "pdfplumber",
        cache_dir:  Optional[Path] = None,
    ) -> None:
        if provider != "ollama":
            raise ConfigurationError(
                f"WholeDocStrategy: unsupported provider {provider!r}. Only 'ollama' is supported."
            )
        self.provider       = provider
        self.model          = model
        self.extractor_name = extractor
        self.cache_dir      = Path(cache_dir) if cache_dir else None
        self._extractor     = get_extractor(extractor)
        self._backend       = get_model(
            provider,
            model,
            host=host,
            max_tokens=16384,
        )

    # ── Extraction, with an optional on-disk cache ────────────────────────────

    def _extract(self, pdf_path: Path) -> list[dict]:
        """Extract blocks, reusing a cached result when one exists.

        Extraction depends only on the PDF and the extractor, not on the model,
        so the cache is keyed by extractor.  Labeling one corpus with three
        models then costs one extraction instead of three — a meaningful saving
        for LightOnOCR, which runs a vision model over every page.
        """
        if self.cache_dir is None:
            logger.info("[wholedoc] extracting from %s via %s …",
                        pdf_path.name, type(self._extractor).__name__)
            return self._extractor.extract(pdf_path)

        cached = self.cache_dir / f"{pdf_path.stem}.json"
        if cached.exists():
            blocks = json.loads(cached.read_text(encoding="utf-8"))
            logger.info("[wholedoc] reusing %d cached block(s) from %s",
                        len(blocks), cached)
            return blocks

        logger.info("[wholedoc] extracting from %s via %s …",
                    pdf_path.name, type(self._extractor).__name__)
        blocks = self._extractor.extract(pdf_path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(blocks, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        logger.info("[wholedoc] extracted blocks cached -> %s", cached)
        return blocks

    # ── Strategy protocol ──────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> list[dict]:
        """Extract and classify all blocks in *pdf_path* in one model call."""
        blocks = self._extract(pdf_path)
        if not blocks:
            logger.warning("[wholedoc] no blocks found in %s", pdf_path.name)
            return []

        if self.extractor_name == "pdfplumber":
            full_text = blocks[0]["text"]
            return self.classify_entire_document(full_text)

        payload = _build_payload(blocks)
        prompt  = _build_wholedoc_prompt(payload)

        logger.info("[wholedoc] sending %d blocks to %s / %s …",
                    len(payload), self.provider, self.model)
        raw    = self._backend.complete(SYSTEM_PROMPT, prompt)
        parsed = parse_llm_json(raw, label=pdf_path.stem)
        return _apply_labels(blocks, parsed)

    def classify_entire_document(self, full_text):
        """Classify a whole document's text in one call — pdfplumber's own path.

        Name and structure copied from the notebook's classify_entire_document():
        same system prompt, same schema, same "one call, whole document" shape,
        returning ``[{"text", "label"}]`` with no block ids, confidence, page,
        or bbox fields, unlike every other extractor's path through this
        strategy. The two differences from the notebook, both intentional:
        ``model`` is this strategy's configured ``self.model`` instead of a
        hardcoded ``"qwen2.5:14b"``, and the call goes through this project's
        own OllamaModel backend (deterministic, temperature=0.0) rather than
        the notebook's raw ``ollama.chat()`` (which had no temperature set and
        measurably varied run to run).
        """
        model = self.model
        print(f"Sending full text to {model}")

        prompt = f"Classify the following text into the required JSON format:\n\n{full_text}"
        raw = self._backend.complete(
            prompt_labels.SYSTEM_PROMPT, prompt, schema=prompt_labels.schema,
        )

        result = parse_llm_json(raw)
        return clean_labeled_output(result)
