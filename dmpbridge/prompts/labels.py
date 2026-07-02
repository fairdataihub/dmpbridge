"""Label definitions and Ollama structured-output schema."""

LABELS = ("title", "section.title", "section.description", "question.text", "answer.text")

OUTPUT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":    {"type": "integer"},
            "label": {"type": "string", "enum": list(LABELS)},
        },
        "required": ["id", "label"],
    },
}
