from pathlib import Path
from typing import Dict, Any, List
import re
import unicodedata
from difflib import SequenceMatcher

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.validation.schema_validator import validate_narrative_json


# Common words removed for Word Capture.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "will"
}


# ---------------------------------------------------------
# Text normalization and basic similarity metrics
# ---------------------------------------------------------

def normalize_eval_text(text: str) -> str:
    """
    Normalize text before evaluation.

    This reduces differences caused by:
    - curly quotes vs straight quotes
    - em dash/en dash vs hyphen
    - extra whitespace
    - uppercase/lowercase
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def tokenize_words(text: str) -> set[str]:
    """
    Convert text into meaningful words for Word Capture.
    """
    text = normalize_eval_text(text)
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def rouge_l_score(extracted: str, reference: str) -> float:
    """
    Approximate ROUGE-L using longest matching text blocks.
    """
    extracted = normalize_eval_text(extracted)
    reference = normalize_eval_text(reference)

    if not extracted or not reference:
        return 0.0

    matcher = SequenceMatcher(None, extracted, reference)
    lcs = sum(block.size for block in matcher.get_matching_blocks())

    return lcs / max(len(reference), 1)


def word_capture(extracted: str, reference: str) -> float:
    """
    Measure how much reference vocabulary appears in extracted text.
    """
    extracted_words = tokenize_words(extracted)
    reference_words = tokenize_words(reference)

    if not reference_words:
        return 1.0

    return len(extracted_words & reference_words) / len(reference_words)


def count_match_score(extracted_count: int, reference_count: int) -> float | None:
    """
    Compare extracted count to reference count.

    Returns None if the reference has no items.
    """
    if reference_count == 0:
        return None

    return min(extracted_count, reference_count) / reference_count


# ---------------------------------------------------------
# JSON access helpers
# ---------------------------------------------------------

def get_narrative_template(dmp_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return narrative.template from either supported JSON shape.
    """
    if "narrative" in dmp_json:
        return dmp_json["narrative"]["template"]

    if "dmp" in dmp_json and "narrative" in dmp_json["dmp"]:
        return dmp_json["dmp"]["narrative"]["template"]

    return {"title": None, "section": []}


def get_sections(dmp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return narrative sections.
    """
    template = get_narrative_template(dmp_json)
    return template.get("section", [])


def get_narrative_title(dmp_json: Dict[str, Any]) -> str:
    """
    Return normalized narrative template title.
    """
    template = get_narrative_template(dmp_json)
    return normalize_eval_text(template.get("title", ""))


def get_section_titles(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Extract normalized section titles.
    """
    return [
        normalize_eval_text(section.get("title", ""))
        for section in get_sections(dmp_json)
        if section.get("title")
    ]


def get_question_texts(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Extract all question.text values.
    """
    question_texts = []

    for section in get_sections(dmp_json):
        for question in section.get("question", []):
            text = question.get("text", "")
            if text:
                question_texts.append(normalize_eval_text(text))

    return question_texts


def get_question_titles(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Extract short question/subsection titles.

    Short question titles are usually things like:
    - A. Types of data
    - Roles & Responsibilities
    """
    titles = []

    for section in get_sections(dmp_json):
        for question in section.get("question", []):
            text = question.get("text", "")

            if text and len(text.split()) <= 12:
                titles.append(normalize_eval_text(text))

    return titles


def get_question_keys(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Get question identifiers for order comparison.

    Prefer question titles. If unavailable, use full question text.
    """
    question_titles = get_question_titles(dmp_json)

    if question_titles:
        return question_titles

    return get_question_texts(dmp_json)


def get_answer_texts(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Extract all answer text values in narrative order.
    """
    answers = []

    for section in get_sections(dmp_json):
        for question in section.get("question", []):
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
    """
    Combine narrative title, section titles, question text, and answer text.
    """
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
# Title, section, question, and answer metric functions
# ---------------------------------------------------------

def narrative_title_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare narrative template titles.
    """
    extracted_title = get_narrative_title(extracted_json)
    reference_title = get_narrative_title(reference_json)

    if not reference_title:
        return None

    return rouge_l_score(extracted_title, reference_title)


def section_title_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare section titles without considering order.
    """
    extracted_titles = set(get_section_titles(extracted_json))
    reference_titles = set(get_section_titles(reference_json))

    if not reference_titles:
        return None

    return len(extracted_titles & reference_titles) / len(reference_titles)


def section_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare section title order.
    """
    extracted_titles = get_section_titles(extracted_json)
    reference_titles = get_section_titles(reference_json)

    if not reference_titles:
        return None

    matches = 0

    for index, ref_title in enumerate(reference_titles):
        if index < len(extracted_titles) and extracted_titles[index] == ref_title:
            matches += 1

    return matches / len(reference_titles)


def question_title_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare short question/subsection titles.
    """
    extracted_titles = set(get_question_titles(extracted_json))
    reference_titles = set(get_question_titles(reference_json))

    if not reference_titles:
        return None

    return len(extracted_titles & reference_titles) / len(reference_titles)


def question_text_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare full question text using best-match ROUGE-L.
    """
    extracted_questions = get_question_texts(extracted_json)
    reference_questions = get_question_texts(reference_json)

    if not reference_questions:
        return None

    if not extracted_questions:
        return 0.0

    scores = []

    for ref_question in reference_questions:
        best_score = 0.0

        for ext_question in extracted_questions:
            score = rouge_l_score(ext_question, ref_question)
            best_score = max(best_score, score)

        scores.append(best_score)

    return sum(scores) / len(scores)


def question_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare question order using question titles or question text.
    """
    extracted_questions = get_question_keys(extracted_json)
    reference_questions = get_question_keys(reference_json)

    if not reference_questions:
        return None

    matches = 0

    for index, ref_question in enumerate(reference_questions):
        if index < len(extracted_questions) and extracted_questions[index] == ref_question:
            matches += 1

    return matches / len(reference_questions)


def answer_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float:
    """
    Compare all answer text using ROUGE-L.
    """
    extracted_text = "\n".join(get_answer_texts(extracted_json))
    reference_text = "\n".join(get_answer_texts(reference_json))

    return rouge_l_score(extracted_text, reference_text)


def answer_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float | None:
    """
    Compare answer order using answer text similarity.
    """
    extracted_answers = get_answer_texts(extracted_json)
    reference_answers = get_answer_texts(reference_json)

    if not reference_answers:
        return None

    matches = 0

    for index, ref_answer in enumerate(reference_answers):
        if index < len(extracted_answers):
            score = rouge_l_score(extracted_answers[index], ref_answer)

            if score >= 0.70:
                matches += 1

    return matches / len(reference_answers)


# ---------------------------------------------------------
# Main evaluation functions
# ---------------------------------------------------------

def format_score(score: float | None) -> float | str:
    """
    Format scores for output table.
    """
    return round(score, 3) if score is not None else "N/A"


def evaluate_one_dmp(
    extracted_json_path: str | Path,
    reference_json_path: str | Path
) -> Dict[str, Any]:
    """
    Evaluate one extracted JSON against one reference JSON.

    Adds alignment_issue to explain cases where content is captured,
    but sections/answers are shifted because of incorrect structure detection.
    """
    extracted_json = load_json(extracted_json_path)
    reference_json = load_json(reference_json_path)

    extracted_text = flatten_narrative_text(extracted_json)
    reference_text = flatten_narrative_text(reference_json)

    validation_errors = validate_narrative_json(extracted_json)

    narrative_title_score = narrative_title_match_score(extracted_json, reference_json)

    section_title_score = section_title_match_score(extracted_json, reference_json)
    section_order_score = section_order_match_score(extracted_json, reference_json)

    question_title_score = question_title_match_score(extracted_json, reference_json)
    question_text_score = question_text_match_score(extracted_json, reference_json)
    question_order_score = question_order_match_score(extracted_json, reference_json)

    answer_score = answer_match_score(extracted_json, reference_json)
    answer_order_score = answer_order_match_score(extracted_json, reference_json)

    section_count_score = count_match_score(
        len(get_sections(extracted_json)),
        len(get_sections(reference_json))
    )

    question_count_score = count_match_score(
        len(get_question_texts(extracted_json)),
        len(get_question_texts(reference_json))
    )

    answer_count_score = count_match_score(
        len(get_answer_texts(extracted_json)),
        len(get_answer_texts(reference_json))
    )

    notes = []
    alignment_issues = []

    if narrative_title_score is None:
        notes.append("No reference narrative title.")

    if section_title_score is None:
        notes.append("No reference section title.")

    if question_title_score is None:
        notes.append("No reference question title.")

    if question_text_score is None:
        notes.append("No reference question text.")

    if answer_order_score is None:
        notes.append("No reference answer text.")

    # Diagnostic: correct section titles exist, but the order is wrong.
    if (
        isinstance(section_title_score, float)
        and isinstance(section_order_score, float)
        and section_title_score >= 0.95
        and section_order_score < 0.50
    ):
        alignment_issues.append("Section titles detected but order shifted.")

    # Diagnostic: answer text is mostly captured, but attached in wrong order.
    if (
        isinstance(answer_score, float)
        and isinstance(answer_order_score, float)
        and answer_score >= 0.95
        and answer_order_score < 0.50
    ):
        alignment_issues.append("Answer text captured but attached to wrong section/order.")

    # Diagnostic: number of sections is correct, but order is wrong.
    if (
        isinstance(section_count_score, float)
        and isinstance(section_order_score, float)
        and section_count_score == 1.0
        and section_order_score < 0.50
    ):
        alignment_issues.append("Section count correct but alignment/order incorrect.")

    return {
        "sample_id": Path(extracted_json_path).stem,
        "word_capture": round(word_capture(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),

        "narrative_title_match": format_score(narrative_title_score),

        "section_title_match": format_score(section_title_score),
        "section_order_match": format_score(section_order_score),
        "section_count_match": format_score(section_count_score),

        "question_title_match": format_score(question_title_score),
        "question_text_match": format_score(question_text_score),
        "question_order_match": format_score(question_order_score),
        "question_count_match": format_score(question_count_score),

        "answer_match": round(answer_score, 3),
        "answer_order_match": format_score(answer_order_score),
        "answer_count_match": format_score(answer_count_score),

        "json_valid": len(validation_errors) == 0,
        "validation_errors": validation_errors,
        "alignment_issue": " ".join(alignment_issues),
        "notes": " ".join(notes),
    }


def evaluate_folder(
    extracted_folder: str | Path,
    reference_folder: str | Path,
    output_path: str | Path | None = None
) -> List[Dict[str, Any]]:
    """
    Evaluate all extracted JSON files in a folder.

    Expected naming:
    extracted: sample1_pdfplumber.json
    reference: sample1_reference.json
    """
    extracted_folder = Path(extracted_folder)
    reference_folder = Path(reference_folder)

    results = []

    for extracted_file in sorted(extracted_folder.glob("*.json")):
        sample_id = extracted_file.stem.replace("_pdfplumber", "")
        reference_file = reference_folder / f"{sample_id}_reference.json"

        if not reference_file.exists():
            results.append({
                "sample_id": extracted_file.stem,
                "word_capture": None,
                "rouge_l": None,

                "narrative_title_match": None,

                "section_title_match": None,
                "section_order_match": None,
                "section_count_match": None,

                "question_title_match": None,
                "question_text_match": None,
                "question_order_match": None,
                "question_count_match": None,

                "answer_match": None,
                "answer_order_match": None,
                "answer_count_match": None,

                "json_valid": False,
                "validation_errors": [],
                "alignment_issue": "",
                "notes": f"Missing reference file: {reference_file}"
            })
            continue

        result = evaluate_one_dmp(
            extracted_json_path=extracted_file,
            reference_json_path=reference_file
        )

        results.append(result)

    if output_path:
        save_json(results, output_path)

    return results


def print_evaluation_table(results: List[Dict[str, Any]]) -> None:
    """
    Print evaluation results as a clean aligned table.
    """
    headers = [
        "sample_id",
        "word_capture",
        "rouge_l",
        "narrative_title_match",
        "section_title_match",
        "section_order_match",
        "section_count_match",
        "question_title_match",
        "question_text_match",
        "question_order_match",
        "question_count_match",
        "answer_match",
        "answer_order_match",
        "answer_count_match",
        "json_valid",
        "notes",
        "alignment_issue",
    ]

    rows = []

    for result in results:
        row = []

        for header in headers:
            value = result.get(header, "")
            row.append(str(value))

        rows.append(row)

    col_widths = []

    for col_index, header in enumerate(headers):
        max_width = len(header)

        for row in rows:
            max_width = max(max_width, len(row[col_index]))

        col_widths.append(max_width)

    header_line = " | ".join(
        header.ljust(col_widths[i])
        for i, header in enumerate(headers)
    )

    print(header_line)

    divider_line = "-+-".join(
        "-" * col_widths[i]
        for i in range(len(headers))
    )

    print(divider_line)

    for row in rows:
        row_line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(headers))
        )

        print(row_line)
        
def print_evaluation_matrix(results: List[Dict[str, Any]]) -> None:
    """
    Print samples as columns and metrics as rows.
    Easier to compare many samples side by side.
    """

    sample_ids = [result.get("sample_id", "") for result in results]

    metrics = [
        "word_capture",
        "rouge_l",
        "narrative_title_match",
        "section_title_match",
        "section_order_match",
        "section_count_match",
        "question_title_match",
        "question_text_match",
        "question_order_match",
        "question_count_match",
        "answer_match",
        "answer_order_match",
        "answer_count_match",
        "json_valid",
        "alignment_issue",
        "notes"
    ]

    headers = ["metric"] + sample_ids

    rows = []

    for metric in metrics:
        row = [metric]

        for result in results:
            row.append(str(result.get(metric, "")))

        rows.append(row)

    col_widths = []

    for col_index, header in enumerate(headers):
        max_width = len(header)

        for row in rows:
            max_width = max(max_width, len(row[col_index]))

        col_widths.append(max_width)

    header_line = " | ".join(
        headers[i].ljust(col_widths[i])
        for i in range(len(headers))
    )

    print(header_line)

    divider_line = "-+-".join(
        "-" * col_widths[i]
        for i in range(len(headers))
    )

    print(divider_line)

    for row in rows:
        row_line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(row))
        )

        print(row_line)