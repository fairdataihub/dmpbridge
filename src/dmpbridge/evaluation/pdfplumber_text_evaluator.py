from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "will"
}


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
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def word_capture_score(extracted_text, reference_text):
    extracted_words = tokenize_words(extracted_text)
    reference_words = tokenize_words(reference_text)

    if not reference_words:
        return 0.0

    return len(extracted_words & reference_words) / len(reference_words)


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

    extracted_text = extracted_txt_path.read_text(encoding="utf-8")
    reference_text = reference_txt_path.read_text(encoding="utf-8")

    extracted_words = tokenize_words(extracted_text)
    reference_words = tokenize_words(reference_text)

    missing_words = sorted(reference_words - extracted_words)

    return {
        "sample_id": extracted_txt_path.stem,
        "word_capture": round(word_capture_score(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),
        "extracted_word_count": len(extracted_words),
        "reference_word_count": len(reference_words),
        "missing_word_count": len(missing_words),
        "missing_words_preview": ", ".join(missing_words[:30]),
        "extracted_line_count": len(extracted_text.splitlines()),
        "reference_line_count": len(reference_text.splitlines()),
    }