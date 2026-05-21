from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter


def clean_repeated_words(text):
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
    text = normalize_eval_text(text)
    return re.findall(r"\b[a-zA-Z]+\b", text)


def word_capture_score(extracted_text, reference_text):
    extracted_words = Counter(tokenize_words(extracted_text))
    reference_words = Counter(tokenize_words(reference_text))

    total_reference_words = sum(reference_words.values())

    if total_reference_words == 0:
        return 0.0

    matched_words = extracted_words & reference_words

    return sum(matched_words.values()) / total_reference_words


def word_precision_recall_f1(extracted_words, reference_words):
    extracted_counter = Counter(extracted_words)
    reference_counter = Counter(reference_words)

    matched_counter = extracted_counter & reference_counter

    correct_word_count = sum(matched_counter.values())
    extracted_word_count = len(extracted_words)
    reference_word_count = len(reference_words)

    precision = correct_word_count / extracted_word_count if extracted_word_count > 0 else 0.0
    recall = correct_word_count / reference_word_count if reference_word_count > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


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

    precision, recall, f1 = word_precision_recall_f1(
        extracted_words,
        reference_words
    )

    return {
        "sample_id": extracted_txt_path.stem,
        "word_capture": round(word_capture_score(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),
        "word_precision": round(precision, 3),
        "word_recall": round(recall, 3),
        "word_f1": round(f1, 3),
        "extracted_word_count": len(extracted_words),
        "reference_word_count": len(reference_words),
        "missing_word_count": missing_word_count,
        "extra_word_count": extra_word_count,
    }