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

    title_style_sections = [
        block for block in blocks
        if is_likely_title_style_section(block)
    ]

    if len(title_style_sections) >= 3:
        return "title_style_sections"

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

    if document_format == "title_style_sections":
        return classify_title_style_line(block)

    return classify_unknown_line(block)


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

    return "content"


def classify_all_caps_line(text: str) -> str:
    if is_all_caps_heading(text):
        return "section"

    if is_numbered_section(text):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    return "content"


def classify_title_style_line(block: Dict) -> str:
    text = block.get("text", "").strip()

    if is_likely_title_style_section(block):
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


def classify_unknown_line(block: Dict) -> str:
    text = block.get("text", "").strip()

    if re.match(r"^Element\s+\d+\s*:", text, flags=re.IGNORECASE):
        return "section"

    if is_numbered_section(text):
        return "section"

    if is_all_caps_heading(text):
        return "section"

    if is_likely_title_style_section(block):
        return "section"

    if re.match(r"^[A-Z]\.\s+", text):
        return "subsection"

    return "content"


def is_numbered_section(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    # Matches:
    # 1. Types of data. The bulk of the data...
    # 2. Data and metadata standards. The PI's...
    inline_match = re.match(r"^\d+\.\s+(.+?)\.\s+.+", text)

    if inline_match:
        heading_part = inline_match.group(1).strip()
        heading_words = heading_part.split()

        # Only accept if the heading part is short
        return 2 <= len(heading_words) <= 10

    # Matches standalone numbered headings:
    # 1. Types of data
    # 2. Data and metadata standards
    words = text.split()

    if len(words) > 14:
        return False

    return re.match(r"^\d+\.\s+[A-Z][A-Za-z]", text) is not None


def is_document_title(text: str) -> bool:
    title_patterns = [
        r"^DATA MANAGEMENT AND SHARING PLAN$",
        r"^DATA MANAGEMENT PLAN$",
        r"^DATA MANAGEMENT$",
        r"^DMP$",
        r"^RESOURCE/DATA SHARING PLAN$"
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


def is_likely_title_style_section(block: Dict) -> bool:
    text = block.get("text", "").strip()
    font_size = block.get("avg_font_size") or 0

    if not text:
        return False

    words = text.split()

    if not (2 <= len(words) <= 10):
        return False

    if font_size < 13:
        return False

    if text.endswith(".") or text.endswith("?"):
        return False

    if looks_like_sentence(text):
        return False

    return is_title_style_heading(text)


def is_title_style_heading(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    words = text.split()

    if not (2 <= len(words) <= 14):
        return False

    if text.endswith(".") or text.endswith(":") or text.endswith("?"):
        return False

    if re.match(r"^\d+[\.\)]\s+", text):
        return False

    if re.match(r"^[A-Z][\.\)]\s+", text):
        return False

    if looks_like_sentence(text):
        return False

    punctuation_count = sum(1 for ch in text if ch in ",;()[]{}")
    if punctuation_count >= 3:
        return False

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if alpha_chars / max(len(text), 1) < 0.6:
        return False

    return True


def looks_like_sentence(text: str) -> bool:
    words = [
        w.strip(".,;:()[]{}").lower()
        for w in text.split()
    ]

    sentence_markers = {
        "we", "our", "they", "this", "these", "those",
        "is", "are", "was", "were", "will", "would",
        "should", "could", "can", "may", "must",
        "be", "been", "being", "have", "has", "had"
    }

    marker_count = sum(1 for word in words if word in sentence_markers)

    if marker_count >= 1 and len(words) >= 6:
        return True

    return False


def detect_structure(blocks: List[Dict]) -> List[Dict]:
    blocks = clean_blocks(blocks)
    document_format = detect_document_format(blocks)

    structured_blocks = []
    seen_non_empty_text = False

    for block in blocks:
        text = block.get("text", "").strip()

        if not text:
            label = "empty"

        elif not seen_non_empty_text:
            # First real text line is usually the DMP/title line.
            label = "document_title"
            seen_non_empty_text = True

        else:
            label = classify_line(block, document_format)

        structured_blocks.append({
            **block,
            "label": label,
            "document_format": document_format
        })

    return structured_blocks