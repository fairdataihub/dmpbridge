from pathlib import Path
from typing import Dict, Any, List
import re
import unicodedata
from difflib import SequenceMatcher

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.validation.schema_validator import validate_narrative_json


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "will"
}


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
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def extract_numbers(text: str) -> set[str]:
    text = normalize_eval_text(text)
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


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


def number_capture(extracted: str, reference: str) -> float:
    extracted_numbers = extract_numbers(extracted)
    reference_numbers = extract_numbers(reference)

    if not reference_numbers:
        return 1.0

    return len(extracted_numbers & reference_numbers) / len(reference_numbers)


def get_narrative_template(dmp_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports your current structure:
    root["narrative"]

    If later you move to official DMPTool structure:
    root["dmp"]["narrative"]
    this function will still work.
    """
    if "narrative" in dmp_json:
        return dmp_json["narrative"]["template"]

    if "dmp" in dmp_json and "narrative" in dmp_json["dmp"]:
        return dmp_json["dmp"]["narrative"]["template"]

    return {"title": None, "section": []}


def flatten_narrative_text(dmp_json: Dict[str, Any]) -> str:
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
    template = get_narrative_template(dmp_json)
    return [
        normalize_eval_text(section.get("title", ""))
        for section in template.get("section", [])
        if section.get("title")
    ]


def section_match_score(extracted_json: Dict[str, Any], reference_json: Dict[str, Any]) -> float:
    extracted_titles = set(get_section_titles(extracted_json))
    reference_titles = set(get_section_titles(reference_json))

    if not reference_titles:
        return 1.0

    return len(extracted_titles & reference_titles) / len(reference_titles)


def answer_match_score(extracted_json: Dict[str, Any], reference_json: Dict[str, Any]) -> float:
    """
    Simple answer quality score using ROUGE-L over all narrative answer text.
    """
    extracted_text = flatten_narrative_text(extracted_json)
    reference_text = flatten_narrative_text(reference_json)

    return rouge_l_score(extracted_text, reference_text)


def evaluate_one_dmp(
    extracted_json_path: str | Path,
    reference_json_path: str | Path
) -> Dict[str, Any]:

    extracted_json = load_json(extracted_json_path)
    reference_json = load_json(reference_json_path)

    extracted_text = flatten_narrative_text(extracted_json)
    reference_text = flatten_narrative_text(reference_json)

    validation_errors = validate_narrative_json(extracted_json)

    scores = {
        "sample_id": Path(extracted_json_path).stem,
        "word_capture": round(word_capture(extracted_text, reference_text), 3),
        "rouge_l": round(rouge_l_score(extracted_text, reference_text), 3),
        "number_capture": round(number_capture(extracted_text, reference_text), 3),
        "section_match": round(section_match_score(extracted_json, reference_json), 3),
        "answer_match": round(answer_match_score(extracted_json, reference_json), 3),
        "json_valid": len(validation_errors) == 0,
        "validation_errors": validation_errors,
    }

    scores["passed"] = (
        scores["word_capture"] >= 0.75
        and scores["rouge_l"] >= 0.75
        and scores["number_capture"] >= 0.75
        and scores["section_match"] >= 0.80
        and scores["answer_match"] >= 0.80
        and scores["json_valid"]
    )

    return scores


def evaluate_folder(
    extracted_folder: str | Path,
    reference_folder: str | Path,
    output_path: str | Path | None = None
) -> List[Dict[str, Any]]:

    extracted_folder = Path(extracted_folder)
    reference_folder = Path(reference_folder)

    results = []

    for extracted_file in sorted(extracted_folder.glob("*.json")):
        # Example:
        # extracted: sample1_pdfplumber.json
        # reference: sample1_reference.json
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
    for result in results:
        print(f"\n{result.get('sample_id')}")

        if "error" in result:
            print(f"   {result['error']}")
            continue

        print(f"  Word Capture:   {result['word_capture']}")
        print(f"  ROUGE-L:        {result['rouge_l']}")
        print(f"  Number Capture: {result['number_capture']}")
        print(f"  Section Match:  {result['section_match']}")
        print(f"  Answer Match:   {result['answer_match']}")
        print(f"  JSON Valid:     {result['json_valid']}")
        print(f"  Status:         {' Passed' if result['passed'] else ' Failed'}")