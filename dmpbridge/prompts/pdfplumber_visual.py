"""System prompt and output schema for the pdfplumber visual-signal path.

Ported from ``notebooks/with-pdfplumber-visual-signals.ipynb``. Unlike the
shared :mod:`dmpbridge.prompts.system` prompt (used by the id-based
per-block payload every other extractor sends), this one classifies a
single whole-document text blob in one call and returns a flat
``[{"text": ..., "label": ...}]`` array with no ``id``/``confidence``
fields — so it needs its own prompt and its own schema, not the shared
``OUTPUT_SCHEMA``.
"""
from .labels import LABELS

VISUAL_SIGNAL_SYSTEM_PROMPT = """
You are a highly accurate data extraction assistant. Your task is to classify lines and paragraphs of text from a Data Management Plan into one of the following five categories:
1. "title": the single main title of the document (appears once, typically at the top, usually short).
2. "section.title": a heading that opens a new top-level section. Often starts with a letter prefix (A., B., C.) or a named phrase like "Element 1:".
3. "section.description": Explanatory or instructional text about what a section covers, NOT phrased as a direct question to the researcher, and NOT the researcher's own response. Typically appears right after a section.title and before any question.text or answer.text.
4. "question.text": A specific question, instruction, or prompt that asks the researcher to address a particular topic. Usually ends in a colon or is phrased as a direct ask.
5. "answer.text": The researcher's actual written response. It's typically narrative text describing what the team will do, has done, or plans to do, usually in first- or third-person about the research team.

FORMATTING MARKERS:
Text wrapped in **double asterisks** was visually emphasized (e.g. bold or larger) in the source PDF. Text wrapped in _underscores_ was italicized. Use these as supporting evidence, not strict rules: a short emphasized phrase at the start of a line followed by longer plain text often functions as a label or question embedded in the same paragraph as its answer — split them and classify separately. Longer emphasized or italicized passages that read as instructional are usually section.description. Not all emphasis indicates a heading; some documents use bold/italics for ordinary emphasis within an answer — use context, not formatting alone. Absence of markers does not mean text can't be a title, section.title, or question.text; some documents don't use bold/italic for structure at all.

RULES:
- Process the entire document. Classify every heading, question, description, and paragraph — do not skip or summarize any of it.
- Reproduce each "text" value verbatim from the source, EXCLUDING the ** and _ formatting markers themselves — strip them out before writing the "text" field.
- If a section has no distinct answer, classify what's there once. Do not invent an empty or placeholder entry to fill an "answer.text" slot.
- Some sections may lack a question.text entirely if the section.title is immediately followed by descriptive content or an answer. Do not force text into "question.text" if it doesn't read as a direct question or prompt.
- Do not hallucinate text. Label only the text from the document.
"""

VISUAL_SIGNAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text":  {"type": "string"},
            "label": {"type": "string", "enum": list(LABELS)},
        },
        "required": ["text", "label"],
    },
}

__all__ = ["VISUAL_SIGNAL_SYSTEM_PROMPT", "VISUAL_SIGNAL_SCHEMA"]
