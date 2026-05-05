from typing import List, Dict
import re


def classify_line(block: Dict) -> str:
    text = block["text"].strip()

    if not text:
        return "empty"

    # Document/template title, not a narrative section
    title_patterns = [
        r"^DATA MANAGEMENT AND SHARING PLAN$",
        r"^DATA MANAGEMENT PLAN$",
        r"^DMP$",
    ]

    for pattern in title_patterns:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return "document_title"

    # Main NIH section: Element 1, Element 2, etc.
    if re.match(r"^Element\s+\d+:", text, flags=re.IGNORECASE):
        return "section"

    # NIMH extra section
    if text.lower().startswith("validation schedule"):
        return "section"

    # Subsections: A., B., C.
    # This must come BEFORE bold/font rules.
    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    # Instruction/question-like prompt
    if text.endswith("?"):
        return "question"

    # Sometimes instruction prompts end with colon
    if text.endswith(":"):
        return "question"

    # Visual fallback, but avoid turning document title into section
    if block.get("is_bold") and (block.get("avg_font_size") or 0) > 11:
        return "section"

    if block.get("is_bold"):
        return "subsection"

    return "content"


def detect_structure(blocks: List[Dict]) -> List[Dict]:
    structured_blocks = []

    for block in blocks:
        label = classify_line(block)

        structured_blocks.append({
            **block,
            "label": label
        })

    return structured_blocks