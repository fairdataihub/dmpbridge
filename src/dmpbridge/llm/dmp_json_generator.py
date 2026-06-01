# src/dmpbridge/llm/dmp_json_generator.py

import json
from pathlib import Path
from json_repair import repair_json


def extract_json_from_text(text):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")

    return text[start:end + 1]


def generate_dmp_json(
    prompt,
    tokenizer,
    model,
    max_new_tokens=8000,
):
    messages = [
        {
            "role": "system",
            "content": "You extract structured JSON from Data Management Plans.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    raw_json = extract_json_from_text(decoded)

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        repaired = repair_json(raw_json)
        return json.loads(repaired)


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)