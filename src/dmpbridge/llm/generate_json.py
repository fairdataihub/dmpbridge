# src/dmpbridge/llm/generate_json.py

import json
from pathlib import Path
from json_repair import repair_json


def extract_json_text(model_output):
    start = model_output.find("{")
    end = model_output.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")

    return model_output[start:end + 1]


def generate_json(llm, prompt):
    response = llm.invoke(prompt)
    text = response.content

    json_text = extract_json_text(text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        return json.loads(repaired)


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)