"""Build dynamic few-shot examples from gold-labeled DMP JSON files."""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
GROUND_TRUTH_DIR = _ROOT / "data/input/ground_truth"

_LABEL_DESCRIPTIONS = {
    "title":               "title",
    "section.title":       "section.title",
    "section.description": "section.description (funder template text — instructs the author what to write)",
    "question.text":       "question.text (sub-question prompt inside a section — asks the author to address a specific topic)",
    "answer.text":         "answer.text (researcher's actual written response — narrative, first-person, describes what the team will do)",
}


def _extract_pairs(path: Path) -> list[tuple[str, str]]:
    """Return flat (text, label) pairs from a gold-labeled DMP JSON file."""
    data     = json.loads(path.read_text(encoding="utf-8"))
    template = data.get("narrative", data).get("template", {})
    pairs: list[tuple[str, str]] = []

    title = template.get("title", "").strip()
    if title:
        pairs.append((title, "title"))

    for section in template.get("section", []):
        sec_title = section.get("title", "").strip()
        if sec_title:
            pairs.append((sec_title, "section.title"))
        if section.get("description"):
            pairs.append((section["description"].strip(), "section.description"))
        for question in section.get("question", []):
            q_text = question.get("text", "").strip()
            # PM rule: when section has no explicit sub-question, the section title is
            # repeated as question.text in the JSON. Skip the duplicate so few-shot
            # examples don't show section headings labeled as question.text.
            if q_text and q_text != sec_title:
                pairs.append((q_text, "question.text"))
            ans = question.get("answer", {})
            ans_text = ""
            if isinstance(ans, dict):
                ans_text = ans.get("json", {}).get("answer", "") or ans.get("text", "")
            if ans_text:
                pairs.append((ans_text.strip(), "answer.text"))

    return pairs


def build_few_shot_examples(
    sample_ids: list[int],
    ground_truth_dir: Path | None = None,
    examples_per_label: int = 2,
    max_text_len: int = 120,
) -> str:
    """Build a few-shot examples block from the specified gold-labeled samples.

    Parameters
    ----------
    sample_ids:
        List of sample indices (1–10) whose gold labels supply the examples.
    ground_truth_dir:
        Directory containing ``sample{N}_dmp.json`` files.
        Defaults to ``data/input/ground_truth``.
    examples_per_label:
        Maximum examples to include per label class.
    max_text_len:
        Truncate example text at this character limit to keep the prompt compact.

    Returns
    -------
    str
        Formatted few-shot block ready to be embedded in the system prompt.
    """
    gdir = Path(ground_truth_dir) if ground_truth_dir else GROUND_TRUTH_DIR

    by_label: dict[str, list[str]] = {lbl: [] for lbl in _LABEL_DESCRIPTIONS}

    for sid in sample_ids:
        gold_path = gdir / f"sample{sid}_dmp.json"
        if not gold_path.exists():
            continue
        for text, label in _extract_pairs(gold_path):
            if label in by_label:
                by_label[label].append(text)

    lines = ["\nEXAMPLES FROM REAL DMP DOCUMENTS:\n"]
    for label, description in _LABEL_DESCRIPTIONS.items():
        seen: set[str] = set()
        picked: list[str] = []
        for t in sorted(by_label[label], key=len):
            t_trunc = t[:max_text_len]
            if t_trunc not in seen:
                seen.add(t_trunc)
                picked.append(t_trunc)
            if len(picked) >= examples_per_label:
                break
        lines.append(f"{description}:")
        for t in picked:
            lines.append(f'  "{t}"')
    lines.append("")
    return "\n".join(lines)


__all__ = ["build_few_shot_examples", "GROUND_TRUTH_DIR"]
