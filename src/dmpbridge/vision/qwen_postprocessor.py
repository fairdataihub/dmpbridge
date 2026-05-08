from pathlib import Path
from typing import List, Dict, Any

from dmpbridge.utils.file_io import load_json, save_json
from dmpbridge.utils.logger import log


def convert_qwen_output_to_structured_blocks(
    qwen_results: List[Dict[str, Any]],
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:
    """
    Convert Qwen2-VL output into the same structured_blocks format
    used by the rule-based pipeline.
    """

    structured_blocks = []

    global_order = 1

    for page_result in qwen_results:
        page_number = page_result.get("page")

        items = page_result.get("items", [])

        for item in items:
            text = item.get("text", "").strip()
            label = item.get("label", "content").strip()

            if not text:
                continue

            structured_blocks.append({
                "source_pdf": source_pdf,
                "page": page_number,
                "line_order": global_order,
                "text": text,
                "label": label,
                "document_format": "qwen_vl",
                "extractor": "qwen_vl",
                "reason": item.get("reason")
            })

            global_order += 1

    return structured_blocks


def save_qwen_structured_blocks(
    qwen_output_path: str | Path,
    output_path: str | Path,
    source_pdf: str | None = None
) -> List[Dict[str, Any]]:
    """
    Load Qwen JSON output, convert it to structured_blocks,
    and save the converted result.
    """

    qwen_output_path = Path(qwen_output_path)
    output_path = Path(output_path)

    qwen_results = load_json(qwen_output_path)

    structured_blocks = convert_qwen_output_to_structured_blocks(
        qwen_results=qwen_results,
        source_pdf=source_pdf
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(structured_blocks, output_path)

    log(f"Saved Qwen structured blocks: {output_path}")

    return structured_blocks