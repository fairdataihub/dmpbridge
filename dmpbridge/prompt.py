"""Centralized prompt definitions for all models and strategies.

Everything the LLM sees lives here:
- LABELS                 : the 5 allowed classification labels
- OUTPUT_SCHEMA          : Ollama JSON schema for structured output enforcement
- SYSTEM_PROMPT          : shared system prompt for all providers and strategies
- build_batch_prompt()   : user-facing prompt for batch inference
- build_wholedoc_prompt(): user-facing prompt for whole-document inference
"""
import json

# ── Labels ────────────────────────────────────────────────────────────────────

LABELS = ("title", "section.title", "section.description", "question.text", "answer.text")

# ── Ollama structured output schema ──────────────────────────────────────────

OUTPUT_SCHEMA = {
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

# ── System prompt ─────────────────────────────────────────────────────────────

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

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_batch_prompt(
    batch: list[dict],
    offset: int,
    context: list[dict] | None = None,
) -> str:
    """Build the user-facing prompt for batch inference with a sliding context window.

    Optionally prepends the last few already-labeled blocks as context so the
    model knows where it is in the document, then lists the blocks to classify.
    """
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
            "PRECEDING BLOCKS (already labeled — use for context only, "
            "do not output labels for these):\n"
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
        + "TO CLASSIFY — return a JSON array with exactly %d entries, one per block:\n"
        % len(batch)
        + json.dumps(payload, ensure_ascii=False)
    )


def build_wholedoc_prompt(payload: list[dict]) -> str:
    """Build the prompt for whole-document inference (all blocks in a single call)."""
    return (
        f"CLASSIFY ALL BLOCKS — return a JSON array with exactly {len(payload)} entries, "
        f"one per block, in the same order:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
