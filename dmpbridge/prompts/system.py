"""Shared system prompt used by all strategies and providers."""

_FEW_SHOT_EXAMPLES = """
EXAMPLES FROM REAL DMP DOCUMENTS:

title:
  "Center for Bio-Inspired Energy Science"
  "CAREER: HIGH-RESOLUTION NMR FOR PARAMAGNETIC SODIUM ELECTRODES"
  "Resource/Data Sharing Plan"

section.title (opens a new top-level section of the document):
  "1. Data sharing and preservation"
  "Element 1: Data Type:"
  "Element 2: Related Tools, Software and/or Code:"
  "Element 3: Standards:"
  "2. Data used in publications"
  "3. Data management resources"
  "Roles and Responsibilities"
  "Types of data"
  "Data storage and preservation of access"
  "Products of Research"

section.description (funder template text — instructs the author what to write):
  "Data management plans should describe whether and how data generated in the course of the proposed research will be shared and preserved."
  "Data management plans should provide a plan for making all research data displayed in publications resulting from the proposed research open, machine-readable, and digitally accessible to the public at the time of publication."
  "Data management plans must protect confidentiality, personal privacy, Personally Identifiable Information and U.S. national, homeland, and economic security."

question.text (sub-question prompt WITHIN an already-open section — always appears after a section.title):
  "A. Types and amount of scientific data expected to be generated in the project:"
  "A. Repository where scientific data and metadata will be archived:"
  "B. Whether access to scientific data will be controlled:"
  "Roles & Responsibilities. For the proposed research, describe who will be responsible for coordinating and ensuring data storage and access."
  "Data Types and Sources. A brief, high-level description of the data to be generated."
  "Data Repositories."
  "Data Volume."

answer.text (researcher's actual written response — narrative, first-person, describes what the team will do):
  "This secondary data analysis project will analyze deidentified data from 48,218 participants from eight studies and the publicly available NHANES cohorts."
  "All data pertaining to any published work will be made available to any person on request. Numerical published data as well as the original raw data files will be freely accessible."
  "Data will be analyzed with custom code by our statistical and computer science team."
"""

_QUESTION_TEXT_RULES = """
SECTION.TITLE vs QUESTION.TEXT — apply these rules in order:

Rule 1 — section.title opens a NEW section of the document:
  Any heading that introduces a new top-level section is section.title.
  This includes numbered elements ("Element 2: Related Tools, Software and/or Code:"),
  numbered sections ("1. Data sharing and preservation"), and named sections
  ("Roles and Responsibilities", "Types of data", "Data storage and preservation of access").
  Even when the heading is the only prompt in the section (no A./B. sub-items follow),
  STILL label it section.title — the downstream converter automatically populates
  question.text from the section title. Do NOT label a section heading question.text
  just because it is the only item in its section.

Rule 2 — question.text is a sub-prompt WITHIN an already-open section:
  question.text only appears after a section.title has already been established for
  the current section. These are specific sub-questions or named sub-topics the author
  must address inside a section. They often begin with a letter prefix (A., B., C.) or
  a bold named topic (e.g. "Roles & Responsibilities.", "Data Types and Sources.",
  "Content and Format.", "Data Repositories.", "Data Volume.").

Rule 3 — When in doubt, choose section.title:
  A heading that could be either is almost certainly a section heading (section.title).
  Prefer section.title unless you can clearly identify an enclosing parent section
  that this block is a sub-topic of.
"""

_PROMPT_HEADER = """You are a classifier for Data Management Plan (DMP) documents.

Label each text block with exactly one of these 5 labels:

- title              : The single main title of the entire document. Appears once, very short.
- section.title      : A numbered or named section heading (e.g. "1. Data sharing", "Element 1: Data Type:", "Products of Research").
- section.description: Funder template text that instructs the author what to write in this section. Uses words like "should", "must", "DMPs should", "provide a plan for". This is NOT written by the researcher.
- question.text      : A sub-question or sub-topic prompt inside a section. Asks the author to address a specific topic. Often starts with a letter prefix (A., B.) or a bold named phrase. Always appears AFTER a section.title in the same section.
- answer.text        : The researcher's actual written response — narrative paragraphs describing what the team will do, has done, or plans to do.

Key distinctions:
- section.description = funder wrote it (requirements/instructions); answer.text = researcher wrote it (actual plans/responses)
- question.text = specific sub-topic prompt inside a section; section.description = overall section requirements
- If a block uses "should" or "must" and sounds like instructions → section.description
- If a block describes what the research team will actually do → answer.text
"""

_CONFIDENCE_INSTRUCTIONS = """
CONFIDENCE SCORE
----------------
For every block, include a "confidence" field: a float from 0.0 to 1.0 reflecting how
certain you are that the chosen label is correct. Base it on structural and lexical clarity:

  1.0   Unambiguous. Only one label is consistent with the text and its structural
        position. No plausible alternative exists.
        Examples: a short document title, a numbered section heading like "1. Data Type",
        funder instructions opening with "Data management plans should …".

  0.8–0.9  High confidence. The chosen label is clearly best, but one minor cue is
           ambiguous — e.g. a prompt-like phrase with no question mark, or a heading
           that could be either section.title or question.text.

  0.6–0.7  Moderate confidence. Two labels are plausible; you chose the most likely one
           based on position, phrasing, and surrounding context. Flag for review.

  0.4–0.5  Low confidence. Evidence is roughly balanced between two or more labels.
           The block is structurally atypical or very short. Definitely flag for review.

  < 0.4   Very uncertain. The text is truncated, mixed-register, or could fit several
          labels. Assign the best-fit label but treat this block as unreliable.

Calibration target: blocks with confidence ≥ 0.9 should be correct at least 90% of
the time. Do not assign high confidence to every block — reserve it for genuinely clear
cases to make the score useful for downstream review prioritisation.
"""

_PROMPT_FOOTER = (
    "You MUST output a JSON array with one entry for EVERY block in the TO CLASSIFY list"
    " — no explanation, no markdown.\n"
    "Each entry must have exactly three fields: \"id\" (integer), \"label\" (string),"
    " and \"confidence\" (float 0.0–1.0).\n"
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
    return (
        _PROMPT_HEADER
        + _CONFIDENCE_INSTRUCTIONS
        + _QUESTION_TEXT_RULES
        + examples
        + _PROMPT_FOOTER
    )


# Module-level constant for backward compatibility — strategies use this by default.
SYSTEM_PROMPT = build_system_prompt()

__all__ = ["SYSTEM_PROMPT", "build_system_prompt"]
