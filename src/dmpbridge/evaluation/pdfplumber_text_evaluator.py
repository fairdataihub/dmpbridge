from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter


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
    Keeps duplicates.
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


def evaluate_pdfplumber_text(extracted_txt_path, reference_txt_path):
    extracted_txt_path = Path(extracted_txt_path)
    reference_txt_path = Path(reference_txt_path)

    extracted_text = extracted_txt_path.read_text(encoding="utf-8", errors="ignore")
    reference_text = reference_txt_path.read_text(encoding="utf-8", errors="ignore")

    extracted_words = tokenize_words(extracted_text)
    reference_words = tokenize_words(reference_text)

    extracted_counter = Counter(extracted_words)
    reference_counter = Counter(reference_words)

    missing_words_counter = reference_counter - extracted_counter
    extra_words_counter = extracted_counter - reference_counter

    missing_words = list(missing_words_counter.elements())
    extra_words = list(extra_words_counter.elements())

    return {
        "sample_id": extracted_txt_path.stem,

        "word_capture": round(word_capture_score(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),

        "extracted_word_count": len(extracted_words),
        "reference_word_count": len(reference_words),

        "missing_word_count": len(missing_words),
        "extra_word_count": len(extra_words),

        "missing_words_preview": ", ".join(missing_words[:30]),
        "extra_words_preview": ", ".join(extra_words[:30]),

        "extracted_line_count": len(extracted_text.splitlines()),
        "reference_line_count": len(reference_text.splitlines()),
    }