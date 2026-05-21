from typing import List, Dict
import re


def remove_repeated_adjacent_words(text: str) -> str:
    """
    Remove duplicated adjacent words caused by PDF extraction.
    """
    words = text.split()

    if not words:
        return text

    cleaned = []

    for word in words:
        # Add the word only if it is not the same as the previous word.
        if not cleaned or word != cleaned[-1]:
            cleaned.append(word)

    return " ".join(cleaned)


def normalize_text(text: str) -> str:
    """
    Normalize one extracted text line.

    This function is intentionally conservative:
    - trims leading/trailing spaces
    - removes duplicated adjacent words
    - normalizes multiple spaces into one space
    """
    if not text:
        return ""

    text = text.strip()

    # Fix pdfplumber duplication like:
    # 'Types Types of of data data'
    text = remove_repeated_adjacent_words(text)

    # Replace repeated whitespace, tabs, or line breaks with one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_blocks(blocks: List[Dict]) -> List[Dict]:

    cleaned_blocks = []

    for block in blocks:
        raw_text = block.get("text", "")
        cleaned_text = normalize_text(raw_text)

        # Skip empty lines after cleaning.
        if not cleaned_text:
            continue

        cleaned_blocks.append({
            **block,
            "raw_text": raw_text,
            "text": cleaned_text
        })

    return cleaned_blocks