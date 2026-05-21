from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path



# Constants


_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u201c": '"',  "\u201d": '"',   # curly double quotes
    "\u2018": "'",  "\u2019": "'",   # curly single quotes / apostrophe
    "\u2013": "-",  "\u2014": "-",   # en-dash, em-dash
}

_WORD_PATTERN = re.compile(r"\b[a-zA-Z]+\b")



# Result model


@dataclass(frozen=True)
class ExtractionEvalResult:
    """
    Evaluation metrics for a single pdfplumber extraction vs. its reference.

    Attributes:
        sample_id:            Stem of the extracted file (e.g. 'dmp_042').
        word_capture:         Fraction of reference words matched (0–1).
        rouge_l:              Character-level ROUGE-L score (0–1).
        word_precision:       Precision of extracted tokens (0–1).
        word_recall:          Recall of extracted tokens (0–1).
        word_f1:              Harmonic mean of precision and recall (0–1).
        extracted_word_count: Total tokens in the extracted text.
        reference_word_count: Total tokens in the reference text.
        missing_word_count:   Tokens in reference but absent from extraction.
        extra_word_count:     Tokens in extraction not present in reference.
    """

    sample_id: str
    word_capture: float
    rouge_l: float
    word_precision: float
    word_recall: float
    word_f1: float
    extracted_word_count: int
    reference_word_count: int
    missing_word_count: int
    extra_word_count: int

    def to_dict(self) -> dict:
        """Return a plain dict — compatible with ``pd.DataFrame``."""
        return asdict(self)



# Text helpers


def clean_repeated_words(text: str) -> str:
    """
    Remove consecutive duplicate tokens within each line.

    Only adjacent duplicates are removed; non-adjacent repetitions
    (e.g. headers appearing on every page) are left untouched.
    """
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        prev: str | None = None
        kept: list[str] = []
        for word in line.split():
            key = re.sub(r"\W", "", word).lower()
            if key != prev:
                kept.append(word)
            prev = key
        cleaned_lines.append(" ".join(kept))

    result = "\n".join(cleaned_lines)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def normalize_text(text: str) -> str:
    """
    Normalize unicode, punctuation, and whitespace; lowercase the result.

    Applied once per text before any metric is computed, so downstream
    functions can assume the text is already normalized.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    for original, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(original, replacement)

    return re.sub(r"\s+", " ", text).lower().strip()


def _tokenize(normalized_text: str) -> list[str]:
    """
    Return alphabetic word tokens from an already-normalized string.

    Accepts pre-normalized input so callers don't pay normalization cost
    twice.  Use ``normalize_text`` before calling this.
    """
    return _WORD_PATTERN.findall(normalized_text)



# Individual metric functions  (accept pre-computed inputs)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _word_capture(
    extracted_counter: Counter,
    reference_counter: Counter,
) -> float:
    """Fraction of reference words matched in the extraction (0–1)."""
    matched = sum((extracted_counter & reference_counter).values())
    total   = sum(reference_counter.values())
    return round(_safe_div(matched, total), 3)


def _precision_recall_f1(
    extracted_words: list[str],
    reference_words: list[str],
) -> tuple[float, float, float]:
    """Word-level precision, recall, and F1 (all rounded to 3 d.p.)."""
    ec = Counter(extracted_words)
    rc = Counter(reference_words)

    correct   = sum((ec & rc).values())
    precision = round(_safe_div(correct, len(extracted_words)), 3)
    recall    = round(_safe_div(correct, len(reference_words)), 3)
    f1        = round(_safe_div(2 * precision * recall, precision + recall), 3)
    return precision, recall, f1


def _rouge_l(extracted: str, reference: str) -> float:
    """
    Character-level ROUGE-L using the longest common subsequence.

    ``autojunk=False`` disables SequenceMatcher's heuristic that can
    silently skip frequent characters, ensuring deterministic results.
    """
    if not extracted or not reference:
        return 0.0

    matcher = SequenceMatcher(None, extracted, reference, autojunk=False)
    lcs     = sum(block.size for block in matcher.get_matching_blocks())
    return round(_safe_div(lcs, len(reference)), 3)



# Public evaluation entry point


def evaluate_pdfplumber_text(
    extracted_txt_path: str | Path,
    reference_txt_path: str | Path,
    clean_text: bool = True,
) -> dict:
    """
    Evaluate a pdfplumber extraction against a reference file.

    Reads both files, optionally deduplicates adjacent repeated words in
    the extracted text, then computes nine evaluation metrics.

    Args:
        extracted_txt_path: Path to the pdfplumber-extracted ``.txt`` file.
        reference_txt_path: Path to the manually curated reference ``.txt``.
        clean_text:         Remove consecutive duplicate words from the
                            extracted text before evaluation (default ``True``).

    Returns:
        ``dict`` of evaluation metrics, suitable for ``pd.DataFrame`` rows.
        Keys match the fields of :class:`ExtractionEvalResult`.
    """
    extracted_path = Path(extracted_txt_path)
    reference_path = Path(reference_txt_path)

    extracted_raw = extracted_path.read_text(encoding="utf-8", errors="ignore")
    reference_raw = reference_path.read_text(encoding="utf-8", errors="ignore")

    if clean_text:
        extracted_raw = clean_repeated_words(extracted_raw)

    # Normalize once; share across every metric to avoid redundant work
    norm_extracted = normalize_text(extracted_raw)
    norm_reference = normalize_text(reference_raw)

    ext_words = _tokenize(norm_extracted)
    ref_words = _tokenize(norm_reference)

    ext_counter = Counter(ext_words)
    ref_counter = Counter(ref_words)

    precision, recall, f1 = _precision_recall_f1(ext_words, ref_words)

    return ExtractionEvalResult(
        sample_id            = extracted_path.stem,
        word_capture         = _word_capture(ext_counter, ref_counter),
        rouge_l              = _rouge_l(norm_extracted, norm_reference),
        word_precision       = precision,
        word_recall          = recall,
        word_f1              = f1,
        extracted_word_count = len(ext_words),
        reference_word_count = len(ref_words),
        missing_word_count   = sum((ref_counter - ext_counter).values()),
        extra_word_count     = sum((ext_counter - ref_counter).values()),
    ).to_dict()