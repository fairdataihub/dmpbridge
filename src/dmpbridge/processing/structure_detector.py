from typing import List, Dict
import re

from dmpbridge.processing.text_cleaner import clean_blocks


def is_noisy_pdf(blocks: List[Dict]) -> bool:
    texts = [
        block.get("text", "").strip()
        for block in blocks
        if block.get("text", "").strip()
    ]

    if not texts:
        return True

    joined = " ".join(texts[:80])

    weird_chars = sum(
        1 for ch in joined
        if not ch.isalnum()
        and not ch.isspace()
        and ch not in ".,;:()[]{}-/&%$#@!?+'\""
    )

    if not joined:
        return True

    weird_ratio = weird_chars / len(joined)

    return weird_ratio > 0.08


def detect_document_format(blocks: List[Dict]) -> str:
    if is_noisy_pdf(blocks):
        return "ocr_messy"

    texts = [
        block.get("text", "").strip()
        for block in blocks
        if block.get("text", "").strip()
    ]

    joined_text = "\n".join(texts[:100])

    if re.search(r"Element\s+\d+\s*:", joined_text, flags=re.IGNORECASE):
        return "nih_element"

    numbered_sections = [
        text for text in texts
        if is_numbered_section(text)
    ]

    if len(numbered_sections) >= 2:
        return "numbered_sections"

    all_caps_sections = [
        text for text in texts
        if is_all_caps_heading(text)
    ]

    if len(all_caps_sections) >= 3:
        return "all_caps_sections"

    title_case_sections = [
        text for text in texts
        if is_title_case_section_heading(text)
    ]

    if len(title_case_sections) >= 3:
        return "title_case_sections"

    return "unknown"


def classify_line(block: Dict, document_format: str) -> str:
    text = block.get("text", "").strip()

    if not text:
        return "empty"

    if is_document_title(text):
        return "document_title"

    if document_format == "ocr_messy":
        return classify_ocr_messy_line(text)

    if document_format == "nih_element":
        return classify_nih_line(block)

    if document_format == "numbered_sections":
        return classify_numbered_line(text)

    if document_format == "all_caps_sections":
        return classify_all_caps_line(text)

    if document_format == "title_case_sections":
        return classify_title_case_line(text)

    return classify_unknown_line(text)


def classify_nih_line(block: Dict) -> str:
    text = block.get("text", "").strip()

    if re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE):
        return "section"

    if text.lower().startswith("validation schedule"):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    if text.endswith("?") or text.endswith(":"):
        return "question"

    if block.get("is_bold") and (block.get("avg_font_size") or 0) > 11:
        return "section"

    if block.get("is_bold"):
        return "subsection"

    return "content"


def classify_numbered_line(text: str) -> str:
    if is_numbered_section(text):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    if re.match(r"^\d+\.\d+\s+", text):
        return "subsection"

    # In numbered documents, avoid aggressive question detection.
    # Period headings are unreliable without layout/model support.
    return "content"


def classify_all_caps_line(text: str) -> str:
    if is_all_caps_heading(text):
        return "section"

    if is_numbered_section(text):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    return "content"


def classify_title_case_line(text: str) -> str:
    if is_title_case_section_heading(text):
        return "section"

    if is_numbered_section(text):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    return "content"


def classify_ocr_messy_line(text: str) -> str:
    if is_numbered_section(text):
        return "section"

    if re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE):
        return "section"

    if is_all_caps_heading(text):
        return "section"

    return "content"


def classify_unknown_line(text: str) -> str:
    if re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE):
        return "section"

    if is_numbered_section(text):
        return "section"

    if is_all_caps_heading(text):
        return "section"

    if is_title_case_section_heading(text):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    return "content"


def is_numbered_section(text: str) -> bool:
    return re.match(r"^\d+\.\s+[A-Z][A-Za-z]", text) is not None


def is_document_title(text: str) -> bool:
    title_patterns = [
        r"^DATA MANAGEMENT AND SHARING PLAN$",
        r"^DATA MANAGEMENT PLAN$",
        r"^DATA MANAGEMENT$",
        r"^DMP$",
        r"^CPS\s+\d{4}$",
        r"^CENTER FOR BIO-INSPIRED ENERGY SCIENCE$",
        r"^UNIV\.?\s+OF\s+CALIFORNIA.*DATA MANAGEMENT PLAN$",
        r"^CAREER:.*$",
    ]

    return any(
        re.match(pattern, text, flags=re.IGNORECASE)
        for pattern in title_patterns
    )


def is_all_caps_heading(text: str) -> bool:
    words = text.split()

    if not (2 <= len(words) <= 12):
        return False

    return text.isupper()


def is_title_case_section_heading(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    words = text.split()

    if not (2 <= len(words) <= 10):
        return False

    if text.endswith(".") or text.endswith(":"):
        return False

    bad_starts = [
        "The ",
        "This ",
        "We ",
        "Data will ",
        "All ",
        "Upon ",
        "First,",
        "Our ",
        "There ",
        "Research records",
        "Software generated",
        "Select videos",
        "Materials will",
        "Stored materials",
    ]

    if any(text.startswith(start) for start in bad_starts):
        return False

    small_words = {
        "and", "or", "of", "for", "to", "in", "on", "with",
        "the", "a", "an", "by"
    }

    valid_words = 0

    for word in words:
        clean = word.strip("():;,-/&")

        if not clean:
            continue

        if clean[0].isupper() or clean.lower() in small_words:
            valid_words += 1

    return valid_words / len(words) >= 0.8


def detect_structure(blocks: List[Dict]) -> List[Dict]:
    blocks = clean_blocks(blocks)
    document_format = detect_document_format(blocks)

    structured_blocks = []

    for block in blocks:
        label = classify_line(block, document_format)

        structured_blocks.append({
            **block,
            "label": label,
            "document_format": document_format
        })

    return structured_blocks