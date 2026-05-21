from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter


def clean_repeated_words(text):
    """
    Remove consecutive duplicate words caused by pdfplumber extraction.

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


def normalize_eval_text(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def tokenize_words(text):
    """
    Total word tokenizer.
    Keeps all words, including stopwords.
    Keeps duplicate words.
    """
    text = normalize_eval_text(text)
    return re.findall(r"\b[a-zA-Z]+\b", text)


def word_capture_score(extracted_text, reference_text):
    """
    Measures how many reference words were captured in extracted text.
    Uses word counts, not unique word sets.
    """
    extracted_words = Counter(tokenize_words(extracted_text))
    reference_words = Counter(tokenize_words(reference_text))

    total_reference_words = sum(reference_words.values())

    if total_reference_words == 0:
        return 0.0

    matched_words = extracted_words & reference_words

    return sum(matched_words.values()) / total_reference_words


def rouge_l_score(extracted_text, reference_text):
    extracted_text = normalize_eval_text(extracted_text)
    reference_text = normalize_eval_text(reference_text)

    if not extracted_text or not reference_text:
        return 0.0

    matcher = SequenceMatcher(None, extracted_text, reference_text)
    lcs = sum(block.size for block in matcher.get_matching_blocks())

    return lcs / max(len(reference_text), 1)


def evaluate_pdfplumber_text(extracted_txt_path, reference_txt_path, clean_text=True):
    extracted_txt_path = Path(extracted_txt_path)
    reference_txt_path = Path(reference_txt_path)

    extracted_text = extracted_txt_path.read_text(encoding="utf-8", errors="ignore")
    reference_text = reference_txt_path.read_text(encoding="utf-8", errors="ignore")

    if clean_text:
        extracted_text = clean_repeated_words(extracted_text)

    extracted_words = tokenize_words(extracted_text)
    reference_words = tokenize_words(reference_text)

    extracted_counter = Counter(extracted_words)
    reference_counter = Counter(reference_words)

    missing_words_counter = reference_counter - extracted_counter
    extra_words_counter = extracted_counter - reference_counter

    missing_word_count = sum(missing_words_counter.values())
    extra_word_count = sum(extra_words_counter.values())

    return {
        "sample_id": extracted_txt_path.stem,
        "word_capture": round(word_capture_score(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),
        "extracted_word_count": len(extracted_words),
        "reference_word_count": len(reference_words),
        "missing_word_count": missing_word_count,
        "extra_word_count": extra_word_count,
    }