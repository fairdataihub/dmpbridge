"""Post-extraction text cleaning for pdfplumber blocks.

Handles word-level artefacts that survive character-level deduplication:
- Consecutive duplicate words produced by some PDF renderers
  (e.g. "Roles Roles and and responsibilities" → "Roles and responsibilities")
- Runs of excess whitespace and blank lines
"""
import re


def clean_repeated_words(text: str) -> str:
    """Remove consecutive duplicate words from a string.

    Some PDFs render each word twice in slightly offset layers so pdfplumber
    captures both, producing repeated words rather than repeated characters.
    """
    if not text:
        return ""

    cleaned_lines = []
    for line in text.splitlines():
        words = line.split()
        cleaned_words = []
        previous_word = None
        for word in words:
            normalized = re.sub(r"[^\w]", "", word).lower()
            if normalized != previous_word:
                cleaned_words.append(word)
            previous_word = normalized
        cleaned_lines.append(" ".join(cleaned_words))

    result = "\n".join(cleaned_lines)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def clean_blocks(blocks: list[dict]) -> list[dict]:
    """Apply text cleaning to every block in place.  Empty blocks are dropped."""
    cleaned = []
    for block in blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        new_block = dict(block)
        text = clean_repeated_words(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        new_block["text"] = text.strip()
        cleaned.append(new_block)
    return cleaned
