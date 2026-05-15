from pathlib import Path
from typing import Dict, Any, List

from jsonschema import Draft202012Validator

from dmpbridge.utils.file_io import load_json
from dmpbridge.utils.logger import log


def validate_against_json_schema(
    dmp_json: Dict[str, Any],
    schema: Dict[str, Any]
) -> List[str]:
    """
    Standard JSON Schema validation.

    This checks:
    - required fields
    - object/list/string/integer types
    - allowed structure defined in the schema
    """
    validator = Draft202012Validator(schema)
    schema_errors = sorted(
        validator.iter_errors(dmp_json),
        key=lambda error: list(error.path)
    )

    errors = []

    for error in schema_errors:
        path = ".".join(str(part) for part in error.path)

        if not path:
            path = "root"

        errors.append(f"{path}: {error.message}")

    return errors


def validate_dmpbridge_rules(dmp_json: Dict[str, Any]) -> List[str]:
    """
    Custom DMPBridge validation.

    This checks project-specific rules that JSON Schema may not fully enforce:
    - section IDs are sequential
    - section order is sequential
    - question order restarts inside each section
    - answers follow the DMPTool textArea answer format
    - answer text is not empty
    """
    errors = []

    if "narrative" not in dmp_json:
        errors.append("Missing top-level key: narrative")
        return errors

    narrative = dmp_json.get("narrative", {})

    if "template" not in narrative:
        errors.append("Missing key: narrative.template")
        return errors

    template = narrative.get("template", {})

    if "section" not in template:
        errors.append("Missing key: narrative.template.section")
        return errors

    sections = template.get("section")

    if not isinstance(sections, list):
        errors.append("narrative.template.section must be a list")
        return errors

    for section_index, section in enumerate(sections, start=1):
        section_path = f"section[{section_index}]"

        if section.get("id") != section_index:
            errors.append(f"{section_path}.id should be {section_index}")

        if section.get("order") != section_index:
            errors.append(f"{section_path}.order should be {section_index}")

        if not section.get("title"):
            errors.append(f"{section_path}.title is missing or empty")

        questions = section.get("question")

        if questions is None:
            errors.append(f"{section_path}.question is missing")
            continue

        if not isinstance(questions, list):
            errors.append(f"{section_path}.question must be a list")
            continue

        for question_order, question in enumerate(questions, start=1):
            question_path = f"{section_path}.question[{question_order}]"

            if question.get("order") != question_order:
                errors.append(
                    f"{question_path}.order should be {question_order}"
                )

            if not question.get("id"):
                errors.append(f"{question_path}.id is missing")

            if not question.get("text"):
                errors.append(f"{question_path}.text is missing or empty")

            answer = question.get("answer")

            if not answer:
                errors.append(f"{question_path}.answer is missing")
                continue

            if not answer.get("id"):
                errors.append(f"{question_path}.answer.id is missing")

            answer_json = answer.get("json")

            if not answer_json:
                errors.append(f"{question_path}.answer.json is missing")
                continue

            if answer_json.get("type") != "textArea":
                errors.append(
                    f"{question_path}.answer.json.type should be 'textArea'"
                )

            meta = answer_json.get("meta", {})

            if meta.get("schemaVersion") != "1.0":
                errors.append(
                    f"{question_path}.answer.json.meta.schemaVersion should be '1.0'"
                )

            answer_text = answer_json.get("answer")

            if not answer_text:
                errors.append(
                    f"{question_path}.answer.json.answer is missing or empty"
                )

    return errors


def validate_narrative_json(
    dmp_json: Dict[str, Any],
    schema: Dict[str, Any] | None = None
) -> List[str]:
    """
    Run validation on one generated DMP JSON.

    If schema is provided:
    - run standard JSON Schema validation

    Always:
    - run custom DMPBridge narrative validation
    """
    errors = []

    if schema is not None:
        schema_errors = validate_against_json_schema(dmp_json, schema)
        errors.extend([f"JSON Schema: {error}" for error in schema_errors])

    custom_errors = validate_dmpbridge_rules(dmp_json)
    errors.extend([f"DMPBridge Rule: {error}" for error in custom_errors])

    return errors


def validate_json_file(
    json_path: str | Path,
    schema_path: str | Path | None = None
) -> List[str]:
    """
    Load and validate one generated JSON file.
    """
    json_path = Path(json_path)
    dmp_json = load_json(json_path)

    schema = None

    if schema_path is not None:
        schema = load_json(Path(schema_path))

    errors = validate_narrative_json(dmp_json, schema=schema)

    if errors:
        log(f"Validation failed for {json_path.name}: {len(errors)} error(s)")
    else:
        log(f"Validation passed for {json_path.name}")

    return errors


def validate_structure_json_folder(
    folder_path: str | Path,
    schema_path: str | Path | None = None
) -> Dict[str, List[str]]:
    """
    Validate all JSON files in data/structure_json.
    """
    folder_path = Path(folder_path)
    results = {}

    for json_file in sorted(folder_path.glob("*.json")):
        errors = validate_json_file(
            json_path=json_file,
            schema_path=schema_path
        )
        results[json_file.name] = errors

    return results


def print_validation_report(results: Dict[str, List[str]]) -> None:
    """
    Print a readable validation summary.
    """
    for filename, errors in results.items():
        print(f"\n{filename}")

        if not errors:
            print("   Passed")
        else:
            print(f"   Failed with {len(errors)} error(s)")
            for error in errors:
                print(f"   - {error}")