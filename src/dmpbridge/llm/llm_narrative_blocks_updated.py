"""
llm_narrative_blocks.py

Purpose:
Convert existing PDFPlumber-extracted DMP blocks into structured narrative blocks
using an LLM-primary, chunked whole-document labeling approach.

Updated strategy:
- Keep the LLM as the primary label proposer.
- Send a slimmer categorical payload to the LLM to reduce prompt bloat.
- Preserve provenance metadata in the final output.
- Retry failed chunks before falling back to content.
- Apply deterministic post-label reconciliation to fix obvious hierarchy violations.
- Optionally assemble a nested hierarchy tree for downstream DMP JSON work.

Final labels:
- document_title
- section
- subsection
- content
"""

import json
import re
from pathlib import Path
from statistics import median
from time import sleep

from json_repair import repair_json

from dmpbridge.processing.text_cleaner import (
    clean_blocks,
    clean_repeated_words,
)


ALLOWED_LABELS = {
    "document_title",
    "section",
    "subsection",
    "content",
}

HEADING_LABELS = {"document_title", "section", "subsection"}


# ------------------------------------------------------------
# Basic normalization helpers
# ------------------------------------------------------------

def normalize_text_simple(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", text.strip()))


def is_punctuation_only(text: str) -> bool:
    return bool(re.match(r"^[^\w]+$", text.strip()))


def safe_float(value):
    try:
        if value is None:
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def get_layout_value(block: dict, key: str):
    return safe_float(block.get(key))


def get_font_size(block: dict) -> float:
    value = (
        block.get("avg_font_size")
        or block.get("font_size")
        or block.get("size")
        or 0
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_body_font_size(blocks: list[dict]) -> float:
    sizes = [
        get_font_size(block)
        for block in blocks
        if get_font_size(block) > 0 and str(block.get("text", "")).strip()
    ]

    return median(sizes) if sizes else 11.0


def is_bold_block(block: dict) -> bool:
    if block.get("is_bold") is True:
        return True

    font_names = block.get("font_names") or []
    font_name = str(block.get("font_name", "")).lower()

    all_fonts = " ".join(str(name).lower() for name in font_names)
    all_fonts = f"{all_fonts} {font_name}"

    return any(
        marker in all_fonts
        for marker in ["bold", "black", "heavy", "semibold", "demibold"]
    )


def get_relative_font_hint(font_size: float, body_font_size: float) -> str:
    if font_size <= 0 or body_font_size <= 0:
        return "unknown"

    ratio = font_size / body_font_size

    if ratio >= 1.20:
        return "much_larger_than_body"
    if ratio >= 1.08:
        return "slightly_larger_than_body"
    if ratio <= 0.90:
        return "smaller_than_body"

    return "similar_to_body"


def get_visual_hint(block: dict, body_font_size: float) -> str:
    text = normalize_text_simple(block.get("text", ""))
    font_size = get_font_size(block)
    relative_font = get_relative_font_hint(font_size, body_font_size)

    style_parts = []

    if is_bold_block(block):
        style_parts.append("bold")

    style_parts.append(relative_font)

    if text.isupper() and len(text.split()) <= 20:
        style_parts.append("uppercase_text")

    return "; ".join(style_parts)


def get_position_hint(index: int, total_blocks: int, page) -> str:
    if total_blocks <= 0:
        return "unknown"

    ratio = index / max(total_blocks - 1, 1)

    if ratio <= 0.05:
        position = "beginning"
    elif ratio <= 0.25:
        position = "early"
    elif ratio <= 0.75:
        position = "middle"
    else:
        position = "late"

    return f"{position}; page={page}"


def get_vertical_gap_before(blocks: list[dict], index: int):
    if index == 0:
        return None

    current_top = get_layout_value(blocks[index], "top")
    previous_bottom = get_layout_value(blocks[index - 1], "bottom")

    if current_top is None or previous_bottom is None:
        return None

    return round(current_top - previous_bottom, 2)


def get_vertical_gap_after(blocks: list[dict], index: int):
    if index + 1 >= len(blocks):
        return None

    current_bottom = get_layout_value(blocks[index], "bottom")
    next_top = get_layout_value(blocks[index + 1], "top")

    if current_bottom is None or next_top is None:
        return None

    return round(next_top - current_bottom, 2)


def sort_blocks_by_reading_order(blocks: list[dict]) -> list[dict]:
    """Sort by page/top/x0 when layout values exist; keep stable order otherwise."""
    indexed = list(enumerate(blocks))

    def sort_key(item):
        original_index, block = item
        page = block.get("page")
        top = get_layout_value(block, "top")
        x0 = get_layout_value(block, "x0")
        line_order = block.get("line_order")

        return (
            page if isinstance(page, (int, float)) else 10**9,
            top if top is not None else 10**9,
            x0 if x0 is not None else 10**9,
            line_order if isinstance(line_order, (int, float)) else 10**9,
            original_index,
        )

    return [block for _, block in sorted(indexed, key=sort_key)]


# ------------------------------------------------------------
# Block compaction
# ------------------------------------------------------------

def compact_blocks_for_labeling(
    pdf_blocks: list[dict],
    sort_reading_order: bool = True,
) -> list[dict]:
    """
    Create compact block records.

    The LLM-facing payload intentionally keeps categorical layout hints and drops
    raw coordinates. Provenance fields are retained internally for final output.
    """
    pdf_blocks = clean_blocks(pdf_blocks)

    if sort_reading_order:
        pdf_blocks = sort_blocks_by_reading_order(pdf_blocks)

    body_font_size = get_body_font_size(pdf_blocks)
    clean_pdf_blocks = []

    for block in pdf_blocks:
        if not isinstance(block, dict):
            continue

        text = normalize_text_simple(block.get("text", ""))

        if not text:
            continue

        if is_page_number(text) or is_punctuation_only(text):
            continue

        clean_block = dict(block)
        clean_block["text"] = clean_repeated_words(text)
        clean_pdf_blocks.append(clean_block)

    compact_blocks = []
    total_blocks = len(clean_pdf_blocks)

    for i, block in enumerate(clean_pdf_blocks):
        text = normalize_text_simple(block.get("text", ""))
        font_size = get_font_size(block)

        compact_blocks.append(
            {
                "block_id": i + 1,
                "text": text,
                "page": block.get("page"),
                "line_order": block.get("line_order"),
                "position_hint": get_position_hint(
                    index=i,
                    total_blocks=total_blocks,
                    page=block.get("page"),
                ),
                "relative_font": get_relative_font_hint(font_size, body_font_size),
                "is_bold": is_bold_block(block),
                "visual_hint": get_visual_hint(block, body_font_size),
                "vertical_gap_before": get_vertical_gap_before(clean_pdf_blocks, i),
                "vertical_gap_after": get_vertical_gap_after(clean_pdf_blocks, i),
                "word_count": len(text.split()),
                # Metadata below is not sent to the LLM prompt, but is useful later.
                "font_size": font_size,
                "body_font_size": body_font_size,
                "x0": get_layout_value(block, "x0"),
                "x1": get_layout_value(block, "x1"),
                "top": get_layout_value(block, "top"),
                "bottom": get_layout_value(block, "bottom"),
            }
        )

    return compact_blocks


def block_for_prompt(block: dict) -> dict:
    """Slim representation sent to the LLM."""
    return {
        "block_id": block["block_id"],
        "text": block["text"],
        "page": block.get("page"),
        "position_hint": block.get("position_hint"),
        "relative_font": block.get("relative_font"),
        "is_bold": block.get("is_bold"),
        "visual_hint": block.get("visual_hint"),
        "vertical_gap_before": block.get("vertical_gap_before"),
        "vertical_gap_after": block.get("vertical_gap_after"),
        "word_count": block.get("word_count"),
    }


# ------------------------------------------------------------
# Chunking
# ------------------------------------------------------------

def make_overlapping_chunks(
    blocks: list[dict],
    chunk_size: int = 45,
    overlap: int = 5,
) -> list[dict]:
    chunks = []

    if not blocks:
        return chunks

    start = 0
    chunk_id = 1

    while start < len(blocks):
        end = min(start + chunk_size, len(blocks))
        context_start = max(0, start - overlap)
        context_end = min(len(blocks), end + overlap)

        chunk_blocks = blocks[context_start:context_end]
        target_ids = [block["block_id"] for block in blocks[start:end]]

        chunks.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "context_start": context_start,
                "context_end": context_end,
                "target_block_ids": target_ids,
                "blocks": chunk_blocks,
            }
        )

        if end >= len(blocks):
            break

        start = end
        chunk_id += 1

    return chunks


# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------

def build_chunk_labeling_prompt(chunk: dict, total_blocks: int) -> str:
    prompt_blocks = [block_for_prompt(block) for block in chunk["blocks"]]

    blocks_text = json.dumps(prompt_blocks, indent=2, ensure_ascii=False)
    target_ids_text = json.dumps(chunk["target_block_ids"], ensure_ascii=False)

    return f"""
You are an expert at reading Data Management Plans (DMPs).

You are given one chunk of a larger DMP document.
The chunk includes context blocks and target blocks.
Use context blocks only to understand hierarchy.
Return labels ONLY for target_block_ids.

Document information:
- total_blocks_in_document: {total_blocks}
- chunk_id: {chunk["chunk_id"]}
- target_block_ids: {target_ids_text}

Available block fields:
- block_id
- text
- page
- position_hint
- relative_font
- is_bold
- visual_hint
- vertical_gap_before
- vertical_gap_after
- word_count

Allowed labels:
- document_title
- section
- subsection
- content

document_title:
The main title of the whole DMP or proposal-related DMP document.
Usually appears near the beginning of the document.
Use this only for the overall document title, not regular headings.
There should normally be only one document_title.

section:
A top-level heading that opens a new major DMP topic, requirement area, or major document part.
Examples include Element 1, Element 2, Element 3, Element 4, Element 5, Element 6,
or major numbered DMP sections such as TYPES OF DATA, Data used in publications,
Policies for access and sharing, or Plans for archiving and preservation of access.

subsection:
A smaller prompt, question, or internal heading that belongs under the most recent section.
Lettered headings such as A., B., and C. are usually subsections when they appear
inside an Element section.

content:
Narrative body text, answers, explanations, guidance, examples, list items,
data descriptions, repository names, institution names, continuation lines,
and anything that is not clearly a title, section, or subsection.

Decision checklist:
Before assigning section, ask whether this block starts a new top-level DMP Element
or major document part. If not, and it is a smaller heading/prompt, use subsection.
If uncertain, use content.

Important instructions:
- Return ONLY valid JSON.
- Do NOT rewrite text.
- Do NOT summarize text.
- Do NOT invent headings.
- Return labels ONLY for target_block_ids.
- Do NOT return context-only block_ids.
- Do NOT omit any target_block_id.
- Do NOT add new block_ids.
- Do NOT change block order.
- If uncertain, label the block as content.

Return format:
[
  {{"block_id": 1, "label": "document_title"}},
  {{"block_id": 2, "label": "section"}},
  {{"block_id": 3, "label": "content"}}
]

Chunk blocks:
{blocks_text}
"""


# ------------------------------------------------------------
# JSON parsing
# ------------------------------------------------------------

def extract_json_array(model_output: str) -> str:
    if not model_output or not str(model_output).strip():
        raise ValueError("LLM returned empty output.")

    model_output = str(model_output).strip()
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        print("\nLLM raw output preview:")
        print(model_output[:2000])
        raise ValueError("No JSON array found in model output.")

    return model_output[start:end + 1]


def parse_llm_labels(model_text: str) -> list[dict]:
    json_text = extract_json_array(model_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        parsed = json.loads(repaired)

    if not isinstance(parsed, list):
        raise ValueError("LLM output JSON is not a list.")

    return parsed


def labels_to_lookup(
    llm_labels: list[dict],
    block_lookup: dict[int, dict],
    allowed_block_ids: set[int] | None = None,
) -> dict[int, str]:
    label_lookup = {}

    for item in llm_labels:
        if not isinstance(item, dict):
            continue

        block_id = item.get("block_id")
        label = normalize_label(item.get("label", ""))

        try:
            block_id = int(block_id)
        except (TypeError, ValueError):
            continue

        if allowed_block_ids is not None and block_id not in allowed_block_ids:
            continue

        if block_id in block_lookup and label:
            label_lookup[block_id] = label

    return label_lookup


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def count_labels(blocks: list[dict]) -> dict:
    counts = {label: 0 for label in ALLOWED_LABELS}

    for block in blocks:
        label = normalize_label(block.get("label", ""))
        if label in counts:
            counts[label] += 1

    return counts


def build_labeling_diagnostics(
    compact_blocks: list[dict],
    label_lookup: dict[int, str],
    chunk_diagnostics: list[dict] | None = None,
    validator_changes: list[dict] | None = None,
    final_blocks: list[dict] | None = None,
) -> dict:
    expected_ids = {block["block_id"] for block in compact_blocks}
    returned_ids = set(label_lookup.keys())
    missing_ids = sorted(expected_ids - returned_ids)

    proposal_counts = {label: 0 for label in ALLOWED_LABELS}
    for label in label_lookup.values():
        if label in proposal_counts:
            proposal_counts[label] += 1

    return {
        "total_blocks": len(compact_blocks),
        "llm_labeled_blocks": len(returned_ids),
        "missing_label_count": len(missing_ids),
        "missing_block_ids": missing_ids,
        "llm_label_counts": proposal_counts,
        "final_label_counts": count_labels(final_blocks or []),
        "validator_change_count": len(validator_changes or []),
        "validator_changes": validator_changes or [],
        "chunk_diagnostics": chunk_diagnostics or [],
    }


# ------------------------------------------------------------
# Deterministic label reconciliation
# ------------------------------------------------------------

def looks_like_element_heading(text: str) -> bool:
    text = normalize_text_simple(text)
    patterns = [
        r"^element\s+\d+\b",
        r"^\d+\.\s+.+",
        r"^\d+\)\s+.+",
    ]
    return any(re.match(pattern, text, flags=re.I) for pattern in patterns)


def looks_like_lettered_subsection(text: str) -> bool:
    return bool(re.match(r"^[A-Z]\.?\s+.+", normalize_text_simple(text)))


def looks_like_heading_candidate(block: dict) -> bool:
    text = normalize_text_simple(block.get("text", ""))
    word_count = len(text.split())

    if word_count == 0 or word_count > 35:
        return False

    if looks_like_element_heading(text) or looks_like_lettered_subsection(text):
        return True

    if text.endswith(":") and word_count <= 25:
        return True

    if block.get("is_bold") and word_count <= 25:
        return True

    if "uppercase_text" in str(block.get("visual_hint", "")) and word_count <= 20:
        return True

    return False


def add_validator_change(
    changes: list[dict],
    block: dict,
    old_label: str,
    new_label: str,
    reason: str,
):
    if old_label == new_label:
        return

    changes.append(
        {
            "block_id": block.get("block_id"),
            "page": block.get("page"),
            "text": block.get("text"),
            "old_label": old_label,
            "new_label": new_label,
            "reason": reason,
        }
    )


def reconcile_labels(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Deterministic validator.

    It does not try to fully replace the LLM. It only fixes high-confidence
    hierarchy problems that are common in chunked labeling.
    """
    corrected = [dict(block) for block in blocks]
    changes = []

    document_title_used = False
    seen_section = False

    for i, block in enumerate(corrected):
        label = normalize_label(block.get("label", "")) or "content"
        text = normalize_text_simple(block.get("text", ""))
        position = str(block.get("position_hint", ""))

        # Only one document title. Late title predictions are almost always wrong.
        if label == "document_title":
            if document_title_used or not ("beginning" in position or "early" in position):
                new_label = "section" if looks_like_heading_candidate(block) else "content"
                add_validator_change(
                    changes,
                    block,
                    label,
                    new_label,
                    "extra_or_late_document_title",
                )
                label = new_label
            else:
                document_title_used = True

        # Lettered headings inside sections should usually be subsection.
        if label == "section" and seen_section and looks_like_lettered_subsection(text):
            add_validator_change(
                changes,
                block,
                label,
                "subsection",
                "lettered_heading_inside_existing_section",
            )
            label = "subsection"

        # A subsection before any section has no parent. Promote only if it looks like a major heading.
        if label == "subsection" and not seen_section:
            new_label = "section" if looks_like_element_heading(text) else "content"
            add_validator_change(
                changes,
                block,
                label,
                new_label,
                "subsection_without_prior_section",
            )
            label = new_label

        # Long narrative blocks should not become headings.
        if label in HEADING_LABELS and len(text.split()) > 45:
            add_validator_change(
                changes,
                block,
                label,
                "content",
                "heading_label_on_long_narrative_block",
            )
            label = "content"

        # Weak heading predictions without any heading evidence become content.
        if label in {"section", "subsection"} and not looks_like_heading_candidate(block):
            add_validator_change(
                changes,
                block,
                label,
                "content",
                "heading_label_without_heading_evidence",
            )
            label = "content"

        corrected[i]["label"] = label

        if label == "section":
            seen_section = True

    return corrected, changes


# ------------------------------------------------------------
# Post-processing and merging
# ------------------------------------------------------------

def postprocess_blocks(
    blocks: list[dict],
    keep_metadata: bool = True,
) -> list[dict]:
    clean_output = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        label = normalize_label(block.get("label", ""))
        text = normalize_text_simple(block.get("text", ""))

        if not label or not text:
            continue

        text = clean_repeated_words(text)

        if is_page_number(text) or is_punctuation_only(text):
            continue

        if keep_metadata:
            clean_block = dict(block)
            clean_block["label"] = label
            clean_block["text"] = text
            clean_output.append(clean_block)
        else:
            clean_output.append({"label": label, "text": text})

    return clean_output


def merge_text(previous_text: str, current_text: str) -> str:
    if previous_text.endswith("-"):
        return previous_text[:-1] + current_text

    if previous_text.endswith((".", ":", ";", "?", "!")):
        return previous_text + "\n\n" + current_text

    return previous_text + " " + current_text


def merge_consecutive_document_title_blocks(blocks: list[dict]) -> list[dict]:
    merged = []

    for block in blocks:
        if (
            merged
            and block["label"] == "document_title"
            and merged[-1]["label"] == "document_title"
        ):
            merged[-1]["text"] = merge_text(merged[-1]["text"], block["text"])
            merged[-1].setdefault("merged_block_ids", [merged[-1].get("block_id")])
            merged[-1]["merged_block_ids"].append(block.get("block_id"))
        else:
            merged.append(dict(block))

    return merged


def merge_consecutive_content_blocks(blocks: list[dict]) -> list[dict]:
    merged = []

    for block in blocks:
        if (
            merged
            and block["label"] == "content"
            and merged[-1]["label"] == "content"
        ):
            merged[-1]["text"] = merge_text(merged[-1]["text"], block["text"])
            merged[-1].setdefault("merged_block_ids", [merged[-1].get("block_id")])
            merged[-1]["merged_block_ids"].append(block.get("block_id"))
        else:
            merged.append(dict(block))

    return merged


# ------------------------------------------------------------
# LLM labeling with retry
# ------------------------------------------------------------

def invoke_llm(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    model_text = response.content if hasattr(response, "content") else str(response)

    if not model_text or not model_text.strip():
        raise ValueError("LLM returned empty response.")

    return model_text


def label_one_chunk(
    llm,
    prompt: str,
    block_lookup: dict[int, dict],
    target_ids: set[int],
) -> dict[int, str]:
    model_text = invoke_llm(llm, prompt)
    llm_labels = parse_llm_labels(model_text)

    return labels_to_lookup(
        llm_labels=llm_labels,
        block_lookup=block_lookup,
        allowed_block_ids=target_ids,
    )


def label_blocks_with_chunked_llm(
    llm,
    compact_blocks: list[dict],
    chunk_size: int = 45,
    overlap: int = 5,
    max_retries: int = 2,
    retry_sleep_seconds: float = 0.0,
) -> tuple[dict[int, str], list[dict]]:
    block_lookup = {block["block_id"]: block for block in compact_blocks}
    chunks = make_overlapping_chunks(compact_blocks, chunk_size=chunk_size, overlap=overlap)

    final_label_lookup = {}
    chunk_diagnostics = []

    for chunk in chunks:
        target_ids = set(chunk["target_block_ids"])
        prompt = build_chunk_labeling_prompt(chunk=chunk, total_blocks=len(compact_blocks))

        errors = []
        chunk_label_lookup = {}
        status = "error"

        for attempt in range(max_retries + 1):
            try:
                chunk_label_lookup = label_one_chunk(
                    llm=llm,
                    prompt=prompt,
                    block_lookup=block_lookup,
                    target_ids=target_ids,
                )

                missing_targets = sorted(target_ids - set(chunk_label_lookup.keys()))

                if not missing_targets:
                    status = "ok"
                    break

                errors.append(
                    {
                        "attempt": attempt + 1,
                        "error_type": "MissingTargets",
                        "error_message": f"Missing target ids: {missing_targets}",
                    }
                )

                status = "partial"

            except Exception as e:
                errors.append(
                    {
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )

            if attempt < max_retries and retry_sleep_seconds > 0:
                sleep(retry_sleep_seconds)

        final_label_lookup.update(chunk_label_lookup)
        missing_targets = sorted(target_ids - set(chunk_label_lookup.keys()))

        chunk_diagnostics.append(
            {
                "chunk_id": chunk["chunk_id"],
                "target_count": len(target_ids),
                "labeled_count": len(chunk_label_lookup),
                "missing_count": len(missing_targets),
                "missing_block_ids": missing_targets,
                "status": status,
                "attempts": len(errors) + (1 if status == "ok" else 0),
                "errors": errors,
            }
        )

    return final_label_lookup, chunk_diagnostics


# ------------------------------------------------------------
# Rebuild and optional tree assembly
# ------------------------------------------------------------

def rebuild_blocks_from_labels(
    compact_blocks: list[dict],
    label_lookup: dict[int, str],
    fallback_label: str = "content",
) -> list[dict]:
    rebuilt_blocks = []

    for block in sorted(compact_blocks, key=lambda b: b["block_id"]):
        text = normalize_text_simple(block.get("text", ""))

        if not text:
            continue

        clean_block = dict(block)
        clean_block["label"] = label_lookup.get(block["block_id"], fallback_label)
        clean_block["text"] = clean_repeated_words(text)
        rebuilt_blocks.append(clean_block)

    return rebuilt_blocks


def assemble_hierarchy_tree(blocks: list[dict]) -> dict:
    """
    Build a nested narrative tree:
    document_title -> sections -> subsections -> content.
    """
    tree = {
        "document_title": None,
        "front_matter": [],
        "sections": [],
    }

    current_section = None
    current_subsection = None

    for block in blocks:
        label = block.get("label")
        item = {
            "text": block.get("text", ""),
            "block_id": block.get("block_id"),
            "page": block.get("page"),
        }

        if label == "document_title":
            tree["document_title"] = item
            current_section = None
            current_subsection = None

        elif label == "section":
            current_section = {
                **item,
                "subsections": [],
                "content": [],
            }
            tree["sections"].append(current_section)
            current_subsection = None

        elif label == "subsection":
            if current_section is None:
                current_section = {
                    "text": "Unassigned Section",
                    "block_id": None,
                    "page": None,
                    "subsections": [],
                    "content": [],
                }
                tree["sections"].append(current_section)

            current_subsection = {
                **item,
                "content": [],
            }
            current_section["subsections"].append(current_subsection)

        else:
            if current_subsection is not None:
                current_subsection["content"].append(item)
            elif current_section is not None:
                current_section["content"].append(item)
            else:
                tree["front_matter"].append(item)

    return tree


# ------------------------------------------------------------
# Main public functions
# ------------------------------------------------------------

def generate_structured_blocks_with_llm_labels_only(
    llm,
    pdf_blocks: list[dict],
    return_diagnostics: bool = False,
    return_tree: bool = False,
    keep_metadata: bool = True,
    chunk_size: int = 45,
    overlap: int = 5,
    max_retries: int = 2,
    sort_reading_order: bool = True,
):
    compact_blocks = compact_blocks_for_labeling(
        pdf_blocks,
        sort_reading_order=sort_reading_order,
    )

    if not compact_blocks:
        empty_result = []
        diagnostics = {
            "total_blocks": 0,
            "llm_labeled_blocks": 0,
            "missing_label_count": 0,
            "missing_block_ids": [],
            "llm_label_counts": {label: 0 for label in ALLOWED_LABELS},
            "final_label_counts": {label: 0 for label in ALLOWED_LABELS},
            "validator_change_count": 0,
            "validator_changes": [],
            "chunk_diagnostics": [],
        }

        if return_tree and return_diagnostics:
            return empty_result, {}, diagnostics
        if return_tree:
            return empty_result, {}
        if return_diagnostics:
            return empty_result, diagnostics
        return empty_result

    label_lookup, chunk_diagnostics = label_blocks_with_chunked_llm(
        llm=llm,
        compact_blocks=compact_blocks,
        chunk_size=chunk_size,
        overlap=overlap,
        max_retries=max_retries,
    )

    rebuilt_blocks = rebuild_blocks_from_labels(
        compact_blocks=compact_blocks,
        label_lookup=label_lookup,
        fallback_label="content",
    )

    rebuilt_blocks, validator_changes = reconcile_labels(rebuilt_blocks)

    blocks = postprocess_blocks(rebuilt_blocks, keep_metadata=keep_metadata)
    blocks = merge_consecutive_document_title_blocks(blocks)
    blocks = merge_consecutive_content_blocks(blocks)

    diagnostics = build_labeling_diagnostics(
        compact_blocks=compact_blocks,
        label_lookup=label_lookup,
        chunk_diagnostics=chunk_diagnostics,
        validator_changes=validator_changes,
        final_blocks=blocks,
    )

    if return_tree:
        tree = assemble_hierarchy_tree(blocks)
        if return_diagnostics:
            return blocks, tree, diagnostics
        return blocks, tree

    if return_diagnostics:
        return blocks, diagnostics

    return blocks


def generate_structured_blocks_with_llm(
    llm,
    pdf_blocks: list[dict],
    return_diagnostics: bool = False,
    return_tree: bool = False,
    keep_metadata: bool = True,
    chunk_size: int = 45,
    overlap: int = 5,
    max_retries: int = 2,
    sort_reading_order: bool = True,
):
    return generate_structured_blocks_with_llm_labels_only(
        llm=llm,
        pdf_blocks=pdf_blocks,
        return_diagnostics=return_diagnostics,
        return_tree=return_tree,
        keep_metadata=keep_metadata,
        chunk_size=chunk_size,
        overlap=overlap,
        max_retries=max_retries,
        sort_reading_order=sort_reading_order,
    )


# ------------------------------------------------------------
# Save helpers
# ------------------------------------------------------------

def save_blocks(blocks: list[dict], output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)


def save_tree(tree: dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)


def save_diagnostics(diagnostics: dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
