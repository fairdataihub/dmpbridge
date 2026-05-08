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
You are analyzing a Data Management Plan PDF page.

Detect only the document structure.

Return valid JSON only with this format:

{
  "page": 1,
  "items": [
    {
      "text": "section or subsection text exactly as shown",
      "label": "section",
      "reason": "brief reason"
    }
  ]
}

Labels must be one of:
- document_title
- section
- subsection
- question
- content

Rules:
- Do not summarize.
- Do not invent text.
- Use text exactly visible in the page image.
- Focus on section headings, subsection headings, and questions.
- Ignore normal paragraph content unless it is clearly a question.
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
            max_new_tokens=1500,
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