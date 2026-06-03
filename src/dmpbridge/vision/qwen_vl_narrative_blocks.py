"""
qwen_vl_narrative_blocks.py

Purpose:
Use PDFPlumber text/layout + rendered PDF page images + Qwen2-VL
to classify DMP blocks as:

- document_title
- section
- subsection
- content
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import pdfplumber
import torch
from json_repair import repair_json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from dmpbridge.processing.text_cleaner import clean_repeated_words


ALLOWED_LABELS = {
    "document_title",
    "section",
    "subsection",
    "content",
}


# ============================================================
# Basic helpers
# ============================================================

def normalize_text_simple(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()
    return label if label in ALLOWED_LABELS else ""


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^\d+$", text.strip()))


def extract_json_array(model_output: str) -> str:
    start = model_output.find("[")
    end = model_output.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in Qwen2-VL output.")

    return model_output[start:end + 1]


# ============================================================
# Qwen2-VL client
# ============================================================

class QwenVLClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

    def invoke_image_prompt(self, image_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text


# ============================================================
# PDF page rendering
# ============================================================

def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    resolution: int = 150,
) -> dict[int, str]:
    """
    Render each PDF page to PNG.

    Returns:
    {
        1: "/path/page_001.png",
        2: "/path/page_002.png"
    }
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_images = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            image_path = output_dir / f"{pdf_path.stem}_page_{page_index:03d}.png"

            page_image = page.to_image(resolution=resolution)
            page_image.save(str(image_path), format="PNG")

            page_images[page_index] = str(image_path)

    return page_images


# ============================================================
# Layout enrichment
# ============================================================

def add_context_and_layout_to_blocks(pdf_blocks: list[dict]) -> list[dict]:
    """
    Add previous/current/next context and layout information.

    Expected input block fields may include:
    - text
    - page
    - x0
    - top
    - x1
    - bottom
    - avg_font_size
    - font_size
    - is_bold
    - label / rule_label
    """
    clean_blocks = []

    for block in pdf_blocks:
        text = normalize_text_simple(block.get("text", ""))

        if not text:
            continue

        if is_page_number(text):
            continue

        clean_blocks.append(
            {
                **block,
                "text": text,
            }
        )

    enriched = []

    for i, block in enumerate(clean_blocks):
        previous_text = ""
        next_text = ""

        if i > 0:
            previous_text = clean_blocks[i - 1]["text"]

        if i + 1 < len(clean_blocks):
            next_text = clean_blocks[i + 1]["text"]

        page = block.get("page", 1)

        enriched.append(
            {
                "block_id": i + 1,
                "page": page,
                "previous_text": previous_text,
                "current_text": block["text"],
                "next_text": next_text,
                "x0": block.get("x0"),
                "top": block.get("top"),
                "x1": block.get("x1"),
                "bottom": block.get("bottom"),
                "font_size": (
                    block.get("avg_font_size")
                    or block.get("font_size")
                    or block.get("size")
                ),
                "is_bold": block.get("is_bold"),
                "rule_label": block.get("label") or block.get("rule_label"),
            }
        )

    return enriched


def group_blocks_by_page(enriched_blocks: list[dict]) -> dict[int, list[dict]]:
    pages = defaultdict(list)

    for block in enriched_blocks:
        page = block.get("page") or 1
        page = int(page)
        pages[page].append(block)

    return dict(pages)


# ============================================================
# Prompt
# ============================================================

def build_qwen_vl_page_prompt(page_blocks: list[dict]) -> str:
    blocks_text = json.dumps(
        page_blocks,
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are reading a Data Management Plan PDF page.

You can see the page image and you are given extracted PDFPlumber blocks
with text, previous/current/next context, and layout coordinates.

Your task:
Classify each block using the page image + block context.

Return ONLY valid JSON.
Do NOT rewrite text.
Do NOT summarize text.
Do NOT invent headings.
Do NOT omit any block_id.

Allowed labels:
- document_title
- section
- subsection
- content

How to reason:

document_title:
- Main title of the DMP.
- Usually at the top of page 1.
- May span multiple consecutive blocks.
- Often centered, uppercase, or visually prominent.

section:
- Major DMP heading.
- Starts a new main topic.
- May be numbered, bold, uppercase, larger font, or standalone.
- Often followed by a long paragraph.
- Examples:
  - Element 1: Data Type:
  - 1. Data sharing and preservation
  - Products of Research
  - Data Format Standards
  - Access and sharing

subsection:
- Smaller prompt or internal heading inside a section.
- Often lettered, decimal-numbered, or prompt-like.
- Examples:
  - A. Types and amount of scientific data expected to be generated in the project:
  - B. Whether access to scientific data will be controlled:
  - 1.1 Data Types

content:
- Paragraphs
- Guidance text
- Answer text
- Wrapped sentence fragments
- List items
- Examples
- Repository names alone
- Institution names alone

Critical reasoning rules:
1. Use the page image for visual layout.
2. Use coordinates to understand whether a block is near the top, standalone, or visually separated.
3. Use previous_text and next_text.
4. If current_text continues previous_text, label it content.
5. If previous_text ends with comma, "and", "or", "of", "to", or "in", current_text is usually content.
6. If current_text starts with "(i)", "(ii)", "(iii)", "-", or "•", label it content.
7. If current_text is short and standalone and next_text is a paragraph, it is probably section.
8. If current_text is paragraph-like, label it content.
9. Do not label wrapped sentence fragments as section.
10. Do not rely on keywords only. Use image + layout + context.

Return format:
[
  {{"block_id": 1, "label": "document_title"}},
  {{"block_id": 2, "label": "section"}},
  {{"block_id": 3, "label": "content"}}
]

Blocks for this page:
{blocks_text}
"""


# ============================================================
# Qwen2-VL page classification
# ============================================================

def classify_page_blocks_with_qwen_vl(
    qwen_client: QwenVLClient,
    page_image_path: str,
    page_blocks: list[dict],
) -> list[dict]:
    prompt = build_qwen_vl_page_prompt(page_blocks)

    model_output = qwen_client.invoke_image_prompt(
        image_path=page_image_path,
        prompt=prompt,
    )

    json_text = extract_json_array(model_output)

    try:
        labels = json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json(json_text)
        labels = json.loads(repaired)

    clean_labels = []

    valid_block_ids = {
        block["block_id"]
        for block in page_blocks
    }

    for item in labels:
        if not isinstance(item, dict):
            continue

        block_id = item.get("block_id")
        label = normalize_label(item.get("label", ""))

        if block_id in valid_block_ids and label:
            clean_labels.append(
                {
                    "block_id": block_id,
                    "label": label,
                }
            )

    return clean_labels


# ============================================================
# Postprocessing
# ============================================================

def merge_consecutive_document_title_blocks(blocks: list[dict]) -> list[dict]:
    merged = []

    for block in blocks:
        if (
            merged
            and block["label"] == "document_title"
            and merged[-1]["label"] == "document_title"
        ):
            merged[-1]["text"] += " " + block["text"]
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
            prev = merged[-1]["text"]
            current = block["text"]

            if prev.endswith("-"):
                merged[-1]["text"] = prev[:-1] + current

            elif prev.endswith((".", ":", ";", "?", "!")):
                merged[-1]["text"] += "\n\n" + current

            else:
                merged[-1]["text"] += " " + current

        else:
            merged.append(dict(block))

    return merged


def light_postprocess_blocks(blocks: list[dict]) -> list[dict]:
    """
    Keep postprocessing light because Qwen2-VL already sees layout.
    """
    output = []
    document_title_seen = False

    for block in blocks:
        label = normalize_label(block.get("label", ""))
        text = normalize_text_simple(block.get("text", ""))

        if not label or not text:
            continue

        text = clean_repeated_words(text)

        if is_page_number(text):
            continue

        if label == "document_title":
            if document_title_seen:
                if output and output[-1]["label"] == "document_title":
                    label = "document_title"
                else:
                    label = "content"
            else:
                document_title_seen = True

        output.append(
            {
                "label": label,
                "text": text,
            }
        )

    output = merge_consecutive_document_title_blocks(output)
    output = merge_consecutive_content_blocks(output)

    return output


# ============================================================
# Main function
# ============================================================

def generate_structured_blocks_with_qwen_vl(
    qwen_client: QwenVLClient,
    pdf_path: str | Path,
    pdf_blocks: list[dict],
    rendered_pages_dir: str | Path,
) -> list[dict]:
    """
    Main DMPBridge function.

    Input:
    - pdf_path: original PDF path
    - pdf_blocks: PDFPlumber-extracted blocks with text/layout
    - rendered_pages_dir: folder for page images

    Output:
    [
      {"label": "document_title", "text": "..."},
      {"label": "section", "text": "..."},
      {"label": "content", "text": "..."}
    ]
    """
    page_images = render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=rendered_pages_dir,
    )

    enriched_blocks = add_context_and_layout_to_blocks(pdf_blocks)
    page_groups = group_blocks_by_page(enriched_blocks)

    label_lookup = {}

    for page_number, page_blocks in page_groups.items():
        page_image_path = page_images.get(page_number)

        if not page_image_path:
            continue

        page_labels = classify_page_blocks_with_qwen_vl(
            qwen_client=qwen_client,
            page_image_path=page_image_path,
            page_blocks=page_blocks,
        )

        for item in page_labels:
            label_lookup[item["block_id"]] = item["label"]

    rebuilt_blocks = []

    for block in enriched_blocks:
        block_id = block["block_id"]
        text = normalize_text_simple(block.get("current_text", ""))

        if not text:
            continue

        label = label_lookup.get(block_id, "content")

        rebuilt_blocks.append(
            {
                "label": label,
                "text": text,
            }
        )

    return light_postprocess_blocks(rebuilt_blocks)


def save_blocks(blocks: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)