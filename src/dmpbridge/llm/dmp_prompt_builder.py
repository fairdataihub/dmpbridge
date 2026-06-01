# src/dmpbridge/llm/dmp_prompt_builder.py

import json


def build_narrative_prompt(dmp_text):
    prompt = f"""
You are an expert Data Management Plan extraction assistant.

Your task is to extract the narrative structure from the DMP text.

Return ONLY valid JSON using this structure:

{{
  "sections": [
    {{
      "section_title": "",
      "question_text": "",
      "answer_text": ""
    }}
  ]
}}

Rules:
1. Extract only narrative DMP content.
2. Do not extract names, dates, IDs, funder metadata, contributors, or administrative metadata unless they are part of the narrative answer.
3. Keep the original wording as much as possible.
4. Use the detected section heading as section_title.
5. If there is no separate question, use the section title as question_text.
6. Put the full paragraph/body text under answer_text.
7. Do not invent missing information.
8. Return ONLY valid JSON.
9. Do not include markdown or explanation.

DMP text:
{dmp_text}
"""
    return prompt