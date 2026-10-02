"""The maDMP schema, used exactly as published, with its $ref pointers resolved."""
import json
from pathlib import Path


def resolve_refs(node, defs):
    """Replace every $ref with the definition it points at. Content unchanged."""
    if isinstance(node, dict):
        if "$ref" in node:
            return resolve_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {k: resolve_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [resolve_refs(v, defs) for v in node]
    return node


def load_schema(path):
    """-> (schema with refs resolved, the same as compact text, number of definitions)."""
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    defs = schema.get("$defs", {})
    full = resolve_refs(schema, defs)
    return full, json.dumps(full, separators=(",", ":")), len(defs)
