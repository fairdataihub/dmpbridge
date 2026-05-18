from pathlib import Path
from typing import Dict, Any, List
import re
import unicodedata
from difflib import SequenceMatcher

from dmpbridge.utils.file_io import load_json, save_json


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "will"
}


# ---------------------------------------------------------
# Text normalization and similarity
# ---------------------------------------------------------

def normalize_eval_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def tokenize_words(text: str) -> set[str]:
    text = normalize_eval_text(text)
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def rouge_l_score(extracted: str, reference: str) -> float:
    extracted = normalize_eval_text(extracted)
    reference = normalize_eval_text(reference)

    if not extracted or not reference:
        return 0.0

    matcher = SequenceMatcher(None, extracted, reference)
    lcs = sum(block.size for block in matcher.get_matching_blocks())

    return lcs / max(len(reference), 1)


def word_capture(extracted: str, reference: str) -> float:
    extracted_words = tokenize_words(extracted)
    reference_words = tokenize_words(reference)

    if not reference_words:
        return 1.0

    return len(extracted_words & reference_words) / len(reference_words)


def format_score(score: float | None) -> float:
    """
    Missing/non-comparable score becomes 0.0.
    """
    return round(score, 3) if score is not None else 0.0


def count_match_score(extracted_count: int, reference_count: int) -> float:
    """
    Compare extracted count to reference count.
    """
    if reference_count == 0:
        return 0.0

    return min(extracted_count, reference_count) / reference_count


# ---------------------------------------------------------
# JSON access helpers
# ---------------------------------------------------------

def get_narrative_template(dmp_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both:
    root["narrative"]["template"]
    root["dmp"]["narrative"]["template"]
    """
    if "narrative" in dmp_json:
        return dmp_json["narrative"]["template"]

    if "dmp" in dmp_json and "narrative" in dmp_json["dmp"]:
        return dmp_json["dmp"]["narrative"]["template"]

    return {"title": None, "section": []}


def get_sections(dmp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    template = get_narrative_template(dmp_json)
    return template.get("section", [])


def get_questions(dmp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return actual question objects.

    This counts questions even when question.text is empty.
    """
    questions = []

    for section in get_sections(dmp_json):
        questions.extend(section.get("question", []))

    return questions


def get_narrative_title(dmp_json: Dict[str, Any]) -> str:
    template = get_narrative_template(dmp_json)
    return normalize_eval_text(template.get("title", ""))


def get_section_titles(dmp_json: Dict[str, Any]) -> List[str]:
    return [
        normalize_eval_text(section.get("title", ""))
        for section in get_sections(dmp_json)
        if section.get("title")
    ]


def get_question_texts(dmp_json: Dict[str, Any]) -> List[str]:
    return [
        normalize_eval_text(question.get("text", ""))
        for question in get_questions(dmp_json)
        if question.get("text", "")
    ]


def get_question_order_keys(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Build question-order keys using section title + question order.

    This avoids confusing question order with answer order when question.text is empty.
    """
    keys = []

    for section in get_sections(dmp_json):
        section_title = normalize_eval_text(section.get("title", ""))

        for question_index, question in enumerate(section.get("question", []), start=1):
            question_order = question.get("order", question_index)
            keys.append(f"{section_title}__q{question_order}")

    return keys


def get_answer_texts(dmp_json: Dict[str, Any]) -> List[str]:
    answers = []

    for question in get_questions(dmp_json):
        answer = (
            question
            .get("answer", {})
            .get("json", {})
            .get("answer", "")
        )

        if isinstance(answer, str) and answer.strip():
            answers.append(normalize_eval_text(answer))

    return answers


def flatten_narrative_text(dmp_json: Dict[str, Any]) -> str:
    parts = []

    title = get_narrative_title(dmp_json)
    if title:
        parts.append(title)

    for section in get_sections(dmp_json):
        if section.get("title"):
            parts.append(section["title"])

        for question in section.get("question", []):
            if question.get("text"):
                parts.append(question["text"])

            answer = (
                question
                .get("answer", {})
                .get("json", {})
                .get("answer", "")
            )

            if isinstance(answer, str):
                parts.append(answer)

    return "\n".join(parts)


# ---------------------------------------------------------
# Metric functions
# ---------------------------------------------------------

def narrative_title_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    reference_title = get_narrative_title(reference_json)
    extracted_title = get_narrative_title(extracted_json)

    if not reference_title:
        return None

    return rouge_l_score(extracted_title, reference_title)


def section_title_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    extracted_titles = set(get_section_titles(extracted_json))
    reference_titles = set(get_section_titles(reference_json))

    if not reference_titles:
        return None

    return len(extracted_titles & reference_titles) / len(reference_titles)


def section_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    extracted_titles = get_section_titles(extracted_json)
    reference_titles = get_section_titles(reference_json)

    if not reference_titles:
        return None

    matches = 0

    for i, ref_title in enumerate(reference_titles):
        if i < len(extracted_titles) and extracted_titles[i] == ref_title:
            matches += 1

    return matches / len(reference_titles)


def question_text_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    extracted_questions = get_question_texts(extracted_json)
    reference_questions = get_question_texts(reference_json)

    if not reference_questions:
        return None

    if not extracted_questions:
        return 0.0

    scores = []

    for ref_question in reference_questions:
        best_score = max(
            rouge_l_score(ext_question, ref_question)
            for ext_question in extracted_questions
        )
        scores.append(best_score)

    return sum(scores) / len(scores)


def question_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare question order using section title + question order keys.
    """
    extracted_keys = get_question_order_keys(extracted_json)
    reference_keys = get_question_order_keys(reference_json)

    if not reference_keys:
        return None

    matches = 0

    for i, ref_key in enumerate(reference_keys):
        if i < len(extracted_keys) and extracted_keys[i] == ref_key:
            matches += 1

    return matches / len(reference_keys)


def answer_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float:
    extracted_text = "\n".join(get_answer_texts(extracted_json))
    reference_text = "\n".join(get_answer_texts(reference_json))

    return rouge_l_score(extracted_text, reference_text)


def answer_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    extracted_answers = get_answer_texts(extracted_json)
    reference_answers = get_answer_texts(reference_json)

    if not reference_answers:
        return None

    matches = 0

    for i, ref_answer in enumerate(reference_answers):
        if i < len(extracted_answers):
            score = rouge_l_score(extracted_answers[i], ref_answer)

            if score >= 0.70:
                matches += 1

    return matches / len(reference_answers)


def build_alignment_issue(metrics: Dict[str, float]) -> str:
    failed_metrics = [
        name for name, value in metrics.items()
        if value < 1.0
    ]

    if failed_metrics:
        return "Failed: " + ", ".join(failed_metrics)

    return "Passed"


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def evaluate_one_dmp(
    extracted_json_path: str | Path,
    reference_json_path: str | Path
) -> Dict[str, Any]:

    extracted_json = load_json(extracted_json_path)
    reference_json = load_json(reference_json_path)

    extracted_text = flatten_narrative_text(extracted_json)
    reference_text = flatten_narrative_text(reference_json)

    narrative_title_score = format_score(
        narrative_title_match_score(extracted_json, reference_json)
    )

    section_title_score = format_score(
        section_title_match_score(extracted_json, reference_json)
    )

    section_order_score = format_score(
        section_order_match_score(extracted_json, reference_json)
    )

    section_count_score = count_match_score(
        len(get_sections(extracted_json)),
        len(get_sections(reference_json))
    )

    question_text_score = format_score(
        question_text_match_score(extracted_json, reference_json)
    )

    question_order_score = format_score(
        question_order_match_score(extracted_json, reference_json)
    )

    question_count_score = count_match_score(
        len(get_questions(extracted_json)),
        len(get_questions(reference_json))
    )

    answer_score = round(
        answer_match_score(extracted_json, reference_json),
        3
    )

    answer_order_score = format_score(
        answer_order_match_score(extracted_json, reference_json)
    )

    answer_count_score = count_match_score(
        len(get_answer_texts(extracted_json)),
        len(get_answer_texts(reference_json))
    )

    alignment_metrics = {
        "narrative_title_match": narrative_title_score,
        "section_title_match": section_title_score,
        "section_order_match": section_order_score,
        "section_count_match": section_count_score,
        "question_order_match": question_order_score,
        "question_count_match": question_count_score,
        "answer_order_match": answer_order_score,
        "answer_count_match": answer_count_score,
    }

    notes = []

    if section_title_score == 1.0 and section_order_score < 1.0:
        notes.append("Section titles captured but section order/alignment is wrong.")

    if answer_score >= 0.95 and answer_order_score < 1.0:
        notes.append("Answer text captured but answer order/alignment is wrong.")

    if question_text_score == 0.0:
        notes.append("Question text unavailable or not matched.")

    return {
        "sample_id": Path(extracted_json_path).stem,
        "word_capture": round(word_capture(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),

        "narrative_title_match": narrative_title_score,

        "section_title_match": section_title_score,
        "section_order_match": round(section_order_score, 3),
        "section_count_match": round(section_count_score, 3),

        "question_text_match": question_text_score,
        "question_order_match": question_order_score,
        "question_count_match": round(question_count_score, 3),

        "answer_match": answer_score,
        "answer_order_match": answer_order_score,
        "answer_count_match": round(answer_count_score, 3),

        "alignment_issue": build_alignment_issue(alignment_metrics),
        "notes": " ".join(notes),
    }


def evaluate_folder(
    extracted_folder: str | Path,
    reference_folder: str | Path,
    output_path: str | Path | None = None
) -> List[Dict[str, Any]]:

    extracted_folder = Path(extracted_folder)
    reference_folder = Path(reference_folder)

    results = []

    for extracted_file in sorted(extracted_folder.glob("*.json")):
        sample_id = extracted_file.stem.replace("_pdfplumber", "")
        reference_file = reference_folder / f"{sample_id}_reference.json"

        if not reference_file.exists():
            results.append({
                "sample_id": extracted_file.stem,
                "word_capture": 0.0,
                "rouge_l": 0.0,
                "narrative_title_match": 0.0,
                "section_title_match": 0.0,
                "section_order_match": 0.0,
                "section_count_match": 0.0,
                "question_text_match": 0.0,
                "question_order_match": 0.0,
                "question_count_match": 0.0,
                "answer_match": 0.0,
                "answer_order_match": 0.0,
                "answer_count_match": 0.0,
                "alignment_issue": "Failed: missing reference file",
                "notes": f"Missing reference file: {reference_file}",
            })
            continue

        results.append(
            evaluate_one_dmp(
                extracted_json_path=extracted_file,
                reference_json_path=reference_file
            )
        )

    if output_path:
        save_json(results, output_path)

    return results