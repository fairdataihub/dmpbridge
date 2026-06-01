# src/dmpbridge/llm/dmp_prompt_builder.py

import json
from pathlib import Path


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_dmp_json_prompt(dmp_markdown, skeleton_json):
    skeleton_text = json.dumps(skeleton_json, indent=2)

    prompt = f"""
You are an expert data management plan extraction assistant.

Your task is to convert the extracted DMP text into valid JSON.

Rules:
1. Use the provided JSON skeleton.
2. Fill only fields supported by the DMP text.
3. Do not invent information.
4. If information is missing, use null, empty string, or empty list depending on the skeleton.
5. Return ONLY valid JSON.
6. Do not include explanation, markdown, or comments.

JSON skeleton:
{skeleton_text}

DMP text:
{dmp_markdown}
"""
    return prompt