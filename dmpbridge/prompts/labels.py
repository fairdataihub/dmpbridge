"""Label definitions and Ollama structured-output schema."""

LABELS = ("title", "section.title", "section.description", "question.text", "answer.text")

OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":         {"type": "integer"},
            "label":      {"type": "string", "enum": list(LABELS)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["id", "label", "confidence"],
    },
}
