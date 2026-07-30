"""Shared system prompt used by all strategies and providers."""

_FEW_SHOT_EXAMPLES = """
EXAMPLES FROM REAL DMP DOCUMENTS:

title:
  "Center for Bio-Inspired Energy Science"
  "CAREER: HIGH-RESOLUTION NMR FOR PARAMAGNETIC SODIUM ELECTRODES"
  "Resource/Data Sharing Plan"

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
  "Related tools, software, and/or code"
  "Roles and Responsibilities"

answer.text (researcher's actual written response — narrative, first-person, describes what the team will do):
  "This secondary data analysis project will analyze deidentified data from 48,218 participants from eight studies and the publicly available NHANES cohorts."
  "All data pertaining to any published work will be made available to any person on request. Numerical published data as well as the original raw data files will be freely accessible."
  "Data will be analyzed with custom code by our statistical and computer science team."
"""

_QUESTION_TEXT_RULES = """
QUESTION.TEXT RULES — apply these before deciding between section.title and question.text:

Rule 1 — Section heading IS the question:
  Some DMP sections contain no explicit sub-question; the heading itself is the prompt.
  When a heading stands alone (no lettered sub-items follow, and it is the only prompt in
  the section), label it question.text rather than section.title.
  Examples:
    "Related tools, software, and/or code"  → question.text
    "Roles and Responsibilities"             → question.text
    "Element 2: Related tools, software, and/or code" → question.text

Rule 2 — Distinguish numbered navigation headings from content prompts:
  Short numbered headings that purely name a section (e.g. "1. Data Type", "Section 3")
  are section.title.
  Headings that name a topic the researcher must address — even without a question mark —
  are question.text.

Rule 3 — When in doubt, prefer question.text over section.title:
  If a block could be either, it is always better to label it question.text so that the
  downstream output always has a question entry for every section. An empty question.text
  is never acceptable.
"""

_PROMPT_HEADER = """You are a classifier for Data Management Plan (DMP) documents.

Label each text block with exactly one of these 5 labels:

- title              : The single main title of the entire document. Appears once, very short.
- section.title      : A numbered or named section heading (e.g. "1. Data sharing", "Element 1: Data Type:", "Products of Research").
- section.description: Funder template text that instructs the author what to write in this section. Uses words like "should", "must", "DMPs should", "provide a plan for". This is NOT written by the researcher.
- question.text      : A sub-question or sub-topic prompt inside a section. Asks the author to address a specific topic. Often starts with a letter prefix (A., B.) or a bold phrase. Also used when the section heading itself IS the only prompt (see rules below).
- answer.text        : The researcher's actual written response — narrative paragraphs describing what the team will do, has done, or plans to do.

Key distinctions:
- section.description = funder wrote it (requirements/instructions); answer.text = researcher wrote it (actual plans/responses)
- question.text = specific sub-topic prompt inside a section; section.description = overall section requirements
- If a block uses "should" or "must" and sounds like instructions → section.description
- If a block describes what the research team will actually do → answer.text
"""

_PROMPT_FOOTER = (
    "You MUST output a JSON array with one entry for EVERY block in the TO CLASSIFY list"
    " — no explanation, no markdown.\n"
)


def build_system_prompt(few_shot_examples: str | None = None) -> str:
    """Return the system prompt, optionally with dynamic few-shot examples.

    Parameters
    ----------
    few_shot_examples:
        Pre-formatted examples block from
        :func:`~dmpbridge.prompts.few_shot.build_few_shot_examples`.
        When ``None``, the default hardcoded examples are used.
    """
    examples = few_shot_examples if few_shot_examples is not None else _FEW_SHOT_EXAMPLES
    return _PROMPT_HEADER + _QUESTION_TEXT_RULES + examples + _PROMPT_FOOTER


# Module-level constant for backward compatibility — strategies use this by default.
SYSTEM_PROMPT = build_system_prompt()

__all__ = ["SYSTEM_PROMPT", "build_system_prompt"]
