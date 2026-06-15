import re
from collections import Counter
from pathlib import Path
from typing import Union


def evaluate_pdfplumber_text(
    extracted_txt_path: Union[str, Path],
    reference_txt_path: Union[str, Path],
    clean_text: bool = True,
) -> dict:
    extracted_txt_path = Path(extracted_txt_path)
    reference_txt_path = Path(reference_txt_path)

    extracted_raw = extracted_txt_path.read_text(encoding="utf-8")
    reference_raw = reference_txt_path.read_text(encoding="utf-8")

    if clean_text:
        extracted_words = _tokenize(extracted_raw)
        reference_words = _tokenize(reference_raw)
    else:
        extracted_words = extracted_raw.split()
        reference_words = reference_raw.split()

    extracted_count = Counter(extracted_words)
    reference_count = Counter(reference_words)

    matched = sum((extracted_count & reference_count).values())

    extracted_word_count = len(extracted_words)
    reference_word_count = len(reference_words)

    extra_word_count = extracted_word_count - matched
    missing_word_count = reference_word_count - matched

    word_capture = matched / reference_word_count if reference_word_count else 0.0
    word_precision = matched / extracted_word_count if extracted_word_count else 0.0
    word_recall = matched / reference_word_count if reference_word_count else 0.0

    if word_precision + word_recall > 0:
        word_f1 = 2 * word_precision * word_recall / (word_precision + word_recall)
    else:
        word_f1 = 0.0

    lcs = _lcs_length(extracted_words, reference_words)
    rouge_l = lcs / reference_word_count if reference_word_count else 0.0

    return {
        "sample_id": extracted_txt_path.stem,
        "word_capture": round(word_capture, 3),
        "rouge_l": round(rouge_l, 3),
        "word_precision": round(word_precision, 3),
        "word_recall": round(word_recall, 3),
        "word_f1": round(word_f1, 3),
        "extracted_word_count": extracted_word_count,
        "reference_word_count": reference_word_count,
        "missing_word_count": missing_word_count,
        "extra_word_count": extra_word_count,
    }


def _tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]
