from typing import List, Dict
import re


def remove_repeated_adjacent_words(text: str) -> str:
    words = text.split()

    if not words:
        return text

    cleaned = []

    for word in words:
        if not cleaned or word != cleaned[-1]:
            cleaned.append(word)

    return " ".join(cleaned)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = remove_repeated_adjacent_words(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_blocks(blocks: List[Dict]) -> List[Dict]:
    cleaned_blocks = []

    for block in blocks:
        raw_text = block.get("text", "")
        cleaned_text = normalize_text(raw_text)

        cleaned_blocks.append({
            **block,
            "raw_text": raw_text,
            "text": cleaned_text
        })

    return cleaned_blocks