"""Convert flat labeled blocks (LLM output) to the DMP Tool narrative JSON schema."""

import json
import re
from pathlib import Path
from typing import Union

_DIGIT_PREFIX  = re.compile(r'^\d')
_ROMAN_PREFIX  = re.compile(r'^\(i{1,4}v?i*\)', re.IGNORECASE)
_INLINE_Q      = re.compile(r'^[A-Z][^.\n]{2,40}\.\s+\w')  # "Label. sentence..."


def preprocess_blocks(blocks: list[dict]) -> list[dict]:
    """
    Fix obvious LLM mis-labels using structural metadata before conversion.

    Rule 1 — Roman numeral list items mis-labeled as section.title → answer.text
        e.g. "(i) the SOL-VIDA Study..." is body text, not a section heading.

    Rule 2 — No title block exists → promote first unnumbered section.title to title
        e.g. "DATA MANAGEMENT" or "Resource/Data Sharing Plan" on page 1, bold,
        with no digit prefix, is almost certainly the document title.

    Rule 3 — Split title: answer.text immediately after a title block, same font,
        before any section → continuation of the title pdfplumber broke across lines.
        e.g. title="CAREER: HIGH-RESOLUTION NMR FOR PARAMAGNETIC SODIUM" followed by
        answer.text="ELECTRODES" (same font, no sections yet) → both are the title.
    """
    # Rule 4: long section.title matching "Label. Description sentence..." → question.text
    #   e.g. "Data Sharing and Data Preservation. A description of the plans..."
    #   Real section titles are short headings; this pattern means the LLM saw an inline question.
    for b in blocks:
        if b.get('label') == 'section.title':
            text = b.get('text', '').strip()
            if len(text) > 40 and _INLINE_Q.match(text) and _DIGIT_PREFIX.match(text) is None:
                b['label'] = 'question.text'

    # Rule 1: roman numeral section.title → answer.text
    for b in blocks:
        if b.get('label') == 'section.title' and _ROMAN_PREFIX.match(b.get('text', '')):
            b['label'] = 'answer.text'

    # Rule 2: no title block → promote first unnumbered section.title to title
    if not any(b.get('label') == 'title' for b in blocks):
        for b in blocks:
            if b.get('label') == 'section.title':
                text = b.get('text', '').strip()
                if not _DIGIT_PREFIX.match(text) and not _ROMAN_PREFIX.match(text):
                    b['label'] = 'title'
                    break

    # Rule 3: split title — answer.text right after a title block, same font + bold style, no sections yet
    for i in range(1, len(blocks)):
        prev, curr = blocks[i - 1], blocks[i]
        if (prev.get('label') == 'title'
                and curr.get('label') == 'answer.text'
                and abs(curr.get('avg_font_size', 0) - prev.get('avg_font_size', 0)) < 0.5
                and curr.get('is_bold') == prev.get('is_bold')
                and not any(b.get('label') in ('section.title', 'section.description', 'question.text')
                            for b in blocks[:i])):
            curr['label'] = 'title'

    return blocks


def to_structured(blocks: list[dict], pdf_url: str = "") -> dict:
    """
    Convert a flat list of labeled blocks into the DMP Tool narrative JSON schema.

    Schema
    ------
    narrative
      download_url          URL to access the PDF version (empty if not known)
      template
        title               Document title (from 'title' block, or generated)
        description         Template description (empty — not extractable from PDF)
        version             "v1"
        section[]
          title             section.title block text
          description       section.description blocks joined with \\n
          order             1-based section index
          question[]
            text            question.text block text
            order           1-based question index within the section
            answer
              json
                type        "textArea" (default for all free-text answers)
                answer      answer.text blocks joined with \\n
                meta
                  schemaVersion  "1.0"

    Notes
    -----
    - id fields are omitted throughout — they cannot be determined from a PDF.
    - answer.text before any question.text  → implicit question with empty text.
    - question.text before any section.title → implicit section with empty title.
    """
    blocks = preprocess_blocks(list(blocks))  # copy so we don't mutate caller's list
    title = ""
    sections: list[dict] = []
    cur_section: dict | None = None
    cur_question: dict | None = None
    sec_order = 0
    q_order = 0

    def _new_section(sec_title: str) -> dict:
        nonlocal sec_order, q_order
        sec_order += 1
        q_order = 0
        return {
            "title": sec_title,
            "description": "",
            "order": sec_order,
            "question": [],
        }

    def _new_question(q_text: str) -> dict:
        nonlocal q_order
        q_order += 1
        return {
            "text": q_text,
            "order": q_order,
            "answer": {
                "json": {
                    "type": "textArea",
                    "answer": "",
                    "meta": {
                        "schemaVersion": "1.0",
                    },
                }
            },
        }

    for block in blocks:
        label = block.get("label", "answer.text")
        text = block.get("text", "").strip()
        if not text:
            continue

        if label == "title":
            if not title:
                title = text
            elif not sections:
                # No sections yet — this is a split title (pdfplumber broke it across lines)
                title = title + " " + text
            else:
                # Title appears mid-document after sections — mis-classification, absorb as answer
                if cur_question is None:
                    if cur_section is None:
                        cur_section = _new_section("")
                        sections.append(cur_section)
                    cur_question = _new_question("")
                    cur_section["question"].append(cur_question)
                existing = cur_question["answer"]["json"]["answer"]
                cur_question["answer"]["json"]["answer"] = (
                    existing + "\n" + text if existing else text
                )

        elif label == "section.title":
            cur_section = _new_section(text)
            sections.append(cur_section)
            cur_question = None

        elif label == "section.description":
            if cur_section is None:
                cur_section = _new_section("")
                sections.append(cur_section)
            if cur_section["description"]:
                cur_section["description"] += "\n" + text
            else:
                cur_section["description"] = text

        elif label == "question.text":
            if cur_section is None:
                cur_section = _new_section("")
                sections.append(cur_section)
            if cur_question is not None and not cur_question["answer"]["json"]["answer"]:
                # same label continuing — aggregate into current question
                cur_question["text"] = (cur_question["text"] + "\n" + text) if cur_question["text"] else text
            else:
                cur_question = _new_question(text)
                cur_section["question"].append(cur_question)

        elif label == "answer.text":
            if cur_question is None:
                if cur_section is None:
                    cur_section = _new_section("")
                    sections.append(cur_section)
                cur_question = _new_question("")
                cur_section["question"].append(cur_question)
            existing = cur_question["answer"]["json"]["answer"]
            cur_question["answer"]["json"]["answer"] = (
                existing + "\n" + text if existing else text
            )

    return {
        "narrative": {
            "download_url": pdf_url,
            "template": {
                "title": title,
                "description": "",
                "version": "v1",
                "section": sections,
            },
        }
    }


def convert_file(
    flat_path: Union[str, Path],
    structured_path: Union[str, Path, None] = None,
    pdf_url: str = "",
) -> dict:
    """
    Load a flat labeled JSON file and return (and optionally save) the structured version.

    Parameters
    ----------
    flat_path       : path to *_<model>.json produced by the pipeline
    structured_path : if given, write the structured JSON here;
                      defaults to <stem>_structured.json next to the input
    pdf_url         : optional URL pointing to the source PDF (stored in download_url)
    """
    flat_path = Path(flat_path)
    blocks = json.loads(flat_path.read_text(encoding="utf-8"))
    structured = to_structured(blocks, pdf_url=pdf_url)

    if structured_path is None:
        structured_path = flat_path.with_name(flat_path.stem + "_structured.json")
    structured_path = Path(structured_path)
    structured_path.write_text(
        json.dumps(structured, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return structured
