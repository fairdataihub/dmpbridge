from pathlib import Path
import json
from typing import List, Dict

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from dmpbridge.utils.logger import log
from dmpbridge.utils.file_io import save_json


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_qwen_model():
    log(f"Loading Qwen2-VL model: {MODEL_ID}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor


def detect_structure_from_image(
    image_path: str | Path,
    model,
    processor
) -> Dict:
    image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")

    prompt = """
You are analyzing one page of a Data Management Plan PDF.

Extract the document structure as a hierarchy.

Return valid JSON only:

{
  "document_title": null,
  "sections": [
    {
      "title": "exact section heading visible on the page",
      "subsections": [
        {
          "title": "exact subsection heading visible on the page"
        }
      ]
    }
  ]
}

Rules:
- Use exact visible text only.
- Do not summarize or rewrite.
- Do not include paragraph/body text.
- Do not include explanations or reasons.
- Do not include page numbers, footers, or running headers.
- If a heading and body text appear on the same line, keep only the heading.
- Main topic headings should become sections.
- Smaller repeated headings under a topic should become subsections.
- If no structure is visible, return:
  {"document_title": null, "sections": []}

How to decide hierarchy:
- A document title describes the whole DMP.
- A section introduces a major topic.
- A subsection is a smaller heading that belongs under the nearest previous section.
- Use visual cues such as size, bolding, spacing, indentation, alignment, and repetition.
- Do not infer missing headings.
- Do not invent text.

Return JSON only.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

    response = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    try:
        return json.loads(response)
    except Exception:
        return {
            "page": None,
            "items": [],
            "raw_response": response,
            "error": "Could not parse Qwen output as JSON"
        }


def detect_structure_from_images(
    image_paths: List[str | Path],
    output_path: str | Path | None = None
) -> List[Dict]:
    model, processor = load_qwen_model()

    results = []

    for page_number, image_path in enumerate(image_paths, start=1):
        log(f"Running Qwen2-VL on page {page_number}: {image_path}")

        page_result = detect_structure_from_image(
            image_path=image_path,
            model=model,
            processor=processor
        )

        page_result["page"] = page_number
        results.append(page_result)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, output_path)
        log(f"Saved Qwen structure output: {output_path}")

    return results