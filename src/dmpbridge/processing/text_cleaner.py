# src/dmpbridge/processing/text_cleaner.py

import re


def clean_repeated_words(text: str) -> str:
    """
    Remove consecutive duplicate words from pdfplumber-extracted text.

    Example:
    'Roles Roles and and responsibilities responsibilities'
    becomes:
    'Roles and responsibilities'
    """

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():
        words = line.split()
        cleaned_words = []

        previous_word = None

        for word in words:
            normalized_word = re.sub(r"[^\w]", "", word).lower()

            if normalized_word != previous_word:
                cleaned_words.append(word)

            previous_word = normalized_word

        cleaned_lines.append(" ".join(cleaned_words))

    cleaned_text = "\n".join(cleaned_lines)

    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def clean_blocks(blocks):
    """
    Clean PDFPlumber blocks before structure detection.
    """

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