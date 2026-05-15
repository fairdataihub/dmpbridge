from typing import List, Dict
import re

from dmpbridge.processing.text_cleaner import clean_blocks


def is_document_title(text: str) -> bool:
    known_titles = [
        r"^DATA MANAGEMENT AND SHARING PLAN$",
        r"^DATA MANAGEMENT PLAN$",
        r"^DATA MANAGEMENT$",
        r"^DMP$",
        r"^RESOURCE/DATA SHARING PLAN$",
    ]

    return any(
        re.match(pattern, text, flags=re.IGNORECASE)
        for pattern in known_titles
    )


def is_nih_element_heading(text: str) -> bool:
    return re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE) is not None


def is_numbered_section(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    # Handles inline heading + answer:
    # "1. Types of data. The bulk of the data..."
    inline_match = re.match(r"^\d+\.\s+(.+?)\.\s+.+", text)

    if inline_match:
        heading_part = inline_match.group(1).strip()
        heading_words = heading_part.split()
        return 2 <= len(heading_words) <= 10

    # Handles standalone heading:
    # "1. Types of data"
    words = text.split()

    if len(words) > 14:
        return False

    return re.match(r"^\d+\.\s+[A-Z][A-Za-z]", text) is not None


def is_all_caps_heading(text: str) -> bool:
    words = text.split()

    if not (2 <= len(words) <= 12):
        return False

    return text.isupper()


def looks_like_sentence(text: str) -> bool:
    words = [
        word.strip(".,;:()[]{}").lower()
        for word in text.split()
    ]

    sentence_markers = {
        "we", "our", "this", "these", "those",
        "is", "are", "was", "were", "will", "would",
        "should", "could", "can", "may", "must",
        "be", "been", "being", "have", "has", "had"
    }

    return len(words) >= 6 and any(word in sentence_markers for word in words)


def is_title_style_section(block: Dict) -> bool:
    text = block.get("text", "").strip()
    font_size = block.get("avg_font_size") or 0

    if not text:
        return False

    words = text.split()

    if not (2 <= len(words) <= 10):
        return False

    if font_size < 13:
        return False

    if text.endswith(".") or text.endswith("?") or text.endswith(":"):
        return False

    if looks_like_sentence(text):
        return False

    return True


def classify_line(block: Dict) -> str:
    text = block.get("text", "").strip()

    if not text:
        return "empty"

    if is_document_title(text):
        return "document_title"

    if is_nih_element_heading(text):
        return "section"

    if is_numbered_section(text):
        return "section"

    if is_all_caps_heading(text):
        return "section"

    if is_title_style_section(block):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    if re.match(r"^\d+\.\d+\s+", text):
        return "subsection"

    return "content"


def detect_structure(blocks: List[Dict]) -> List[Dict]:
    """
    Main structure detection function.

    This is a simple rule-based baseline:
    - first non-empty line is document_title
    - numbered/NIH/all-caps/large short headings become sections
    - A. or 1.1 style lines become subsections
    - everything else becomes content
    """

    blocks = clean_blocks(blocks)

    structured_blocks = []
    seen_non_empty_text = False

    for block in blocks:
        text = block.get("text", "").strip()

        if not text:
            label = "empty"

        elif not seen_non_empty_text:
            label = "document_title"
            seen_non_empty_text = True

        else:
            label = classify_line(block)

        structured_blocks.append({
            **block,
            "label": label
        })

    return structured_blocks