from pathlib import Path
from typing import Dict, Any, List
import re
import unicodedata
from difflib import SequenceMatcher

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.validation.schema_validator import validate_narrative_json


# Common words removed when calculating Word Capture.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "will"
}


def normalize_eval_text(text: str) -> str:
    """
    Normalize text before evaluation so small formatting differences do not
    unfairly reduce scores.
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
    Convert text into a set of meaningful words for Word Capture.
    """
    text = normalize_eval_text(text)
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def rouge_l_score(extracted: str, reference: str) -> float:
    """
    Approximate ROUGE-L using longest matching blocks.

    Measures how similar the extracted text sequence is to the reference text.
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
    Measure how much reference vocabulary was captured in the extracted text.
    """
    extracted_words = tokenize_words(extracted)
    reference_words = tokenize_words(reference)

    if not reference_words:
        return 1.0

    return len(extracted_words & reference_words) / len(reference_words)


def get_narrative_template(dmp_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return narrative.template from either supported JSON shape.
    """
    if "narrative" in dmp_json:
        return dmp_json["narrative"]["template"]

    if "dmp" in dmp_json and "narrative" in dmp_json["dmp"]:
        return dmp_json["dmp"]["narrative"]["template"]

    return {"title": None, "section": []}


def flatten_narrative_text(dmp_json: Dict[str, Any]) -> str:
    """
    Combine title, section titles, question text, and answers into one string.

    Used for global metrics:
    - Word Capture
    - ROUGE-L
    - Answer Match
    """
    template = get_narrative_template(dmp_json)

    parts = []

    if template.get("title"):
        parts.append(template["title"])

    for section in template.get("section", []):
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


def get_section_titles(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Extract normalized section titles from narrative.template.section.
    """
    template = get_narrative_template(dmp_json)

    return [
        normalize_eval_text(section.get("title", ""))
        for section in template.get("section", [])
        if section.get("title")
    ]


def section_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float:
    """
    Check whether the extracted JSON found the same section titles as reference.

    This ignores order.
    """
    extracted_titles = set(get_section_titles(extracted_json))
    reference_titles = set(get_section_titles(reference_json))

    if not reference_titles:
        return 1.0

    return len(extracted_titles & reference_titles) / len(reference_titles)


def section_order_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float:
    """
    Check whether sections appear in the correct order.
    """
    extracted_titles = get_section_titles(extracted_json)
    reference_titles = get_section_titles(reference_json)

    if not reference_titles:
        return 1.0

    matches = 0

    for index, ref_title in enumerate(reference_titles):
        if index < len(extracted_titles) and extracted_titles[index] == ref_title:
            matches += 1

    return matches / len(reference_titles)


def get_question_texts(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Extract all question.text values.

    This is important because an output can capture all text but put it in
    the wrong place, such as using section title as question text.
    """
    template = get_narrative_template(dmp_json)

    question_texts = []

    for section in template.get("section", []):
        for question in section.get("question", []):
            text = question.get("text", "")
            if text:
                question_texts.append(normalize_eval_text(text))

    return question_texts


def question_text_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float:
    """
    Compare extracted question text to reference question text.

    Uses best-match ROUGE-L for each reference question.
    """
    extracted_questions = get_question_texts(extracted_json)
    reference_questions = get_question_texts(reference_json)

    if not reference_questions:
        return 1.0

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


def answer_match_score(
    extracted_json: Dict[str, Any],
    reference_json: Dict[str, Any]
) -> float:
    """
    Compare overall narrative answer quality.

    This is intentionally broad, so it should be interpreted together with
    Question Text Match and Section Match.
    """
    extracted_text = flatten_narrative_text(extracted_json)
    reference_text = flatten_narrative_text(reference_json)

    return rouge_l_score(extracted_text, reference_text)


def evaluate_one_dmp(
    extracted_json_path: str | Path,
    reference_json_path: str | Path
) -> Dict[str, Any]:
    """
    Evaluate one extracted DMP JSON against one manually created reference JSON.
    """

    extracted_json = load_json(extracted_json_path)
    reference_json = load_json(reference_json_path)

    extracted_text = flatten_narrative_text(extracted_json)
    reference_text = flatten_narrative_text(reference_json)

    validation_errors = validate_narrative_json(extracted_json)

    scores = {
        "sample_id": Path(extracted_json_path).stem,
        "word_capture": round(word_capture(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),
        "section_match": round(section_match_score(extracted_json, reference_json), 3),
        "section_order_match": round(section_order_match_score(extracted_json, reference_json), 3),
        "question_text_match": round(
            question_text_match_score(extracted_json, reference_json), 3
        ),
        "answer_match": round(answer_match_score(extracted_json, reference_json), 3),
        "json_valid": len(validation_errors) == 0,
        "validation_errors": validation_errors,
    }

    scores["passed"] = (
        scores["word_capture"] >= 0.75
        and scores["rouge_l"] >= 0.75
        and scores["section_match"] >= 0.80
        and scores["section_order_match"] >= 0.80
        and scores["question_text_match"] >= 0.80
        and scores["answer_match"] >= 0.80
        and scores["json_valid"]
    )

    return scores


def evaluate_folder(
    extracted_folder: str | Path,
    reference_folder: str | Path,
    output_path: str | Path | None = None
) -> List[Dict[str, Any]]:
    """
    Evaluate all extracted JSON files against reference JSON files.

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
                "passed": False,
                "error": f"Missing reference file: {reference_file}"
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


def print_evaluation_report(results: List[Dict[str, Any]]) -> None:
    """
    Print evaluation results in a readable format.
    """
    for result in results:
        print(f"\n{result.get('sample_id')}")

        if "error" in result:
            print(f"   {result['error']}")
            continue

        print(f"  Word Capture:        {result['word_capture']}")
        print(f"  ROUGE-L:             {result['rouge_l']}")
        print(f"  Section Match:       {result['section_match']}")
        print(f"  Section Order Match: {result['section_order_match']}")
        print(f"  Question Text Match: {result['question_text_match']}")
        print(f"  Answer Match:        {result['answer_match']}")
        print(f"  JSON Valid:          {result['json_valid']}")
        print(f"  Status:              {'Passed' if result['passed'] else 'Failed'}")