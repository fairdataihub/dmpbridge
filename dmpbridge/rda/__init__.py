"""RDA maDMP extraction: a plan's text -> one RDA maDMP 1.2 JSON object.

The plan's text comes from the same stage-1 extraction as the narrative pipeline
(pdfplumber, LightOnOCR fallback for an unreadable PDF); one model call with the
RDA prompt (``PROMPT_VERSION``) and the maDMP schema then produces the RDA JSON. See
:func:`extract_rda` and the ``dmpbridge-rda`` command.

    from dmpbridge.rda import extract_rda
    result = extract_rda(plan_text, model="llama3.1:8b")
"""
from .extract import (
    PROMPT_VERSION, RdaExtractionError, extract_rda, load_schema, make_prompt, merge, system_prompt,
)

__all__ = ["PROMPT_VERSION", "RdaExtractionError", "extract_rda", "load_schema", "make_prompt",
           "merge", "system_prompt"]
