"""Label the DMP samples from Docling's native document hierarchy alone — no LLM.

The question this answers: how much of the five-label DMP structure is already
present in Docling's ``DoclingDocument`` (item labels, header levels, list
groups, page/bbox provenance, furniture), before any model reads the text?
It writes stages 2–4 for two rule sets into ordinary tag directories, so
``scripts/compare_results.py`` scores them beside every model run.

Rule sets
---------
``structure``   — item labels and provenance only, nothing read from the text:
                  the topmost section_header on page 1 -> title; every other
                  section_header -> section.title; text / list_item -> answer.text;
                  furniture (page headers/footers) dropped.
``structure+``  — the same, plus one text cue a rule-based system would add:
                  a short item ending in ':' or '?' -> question.text. This is
                  the ceiling for "structure plus punctuation" without fonts.

Neither rule set can see bold, italic or underline — Docling does not carry
them (see notebooks/investigation-docling-native-json-sample2.ipynb) — so
section.description is never predicted: nothing in the hierarchy marks it.

    python scripts/docling_structure_baseline.py            # samples 1-10
    python scripts/docling_structure_baseline.py --start 2 --end 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from dmpbridge.core import paths as P
from dmpbridge.core.converter import to_structured
from dmpbridge.evaluation.annotation_rules import apply_new_annotation_rules

EXTRACTOR = "docling"
RULE_SETS = {"structure": False, "structure+": True}
QUESTION_CUE = re.compile(r"[:?]\s*$")
QUESTION_MAX_CHARS = 160


def tag_for(rule_set: str) -> str:
    """One tag per rule set, in the usual <model>_<extractor>_<strategy> shape."""
    return P.make_tag(rule_set.replace("+", "-plus"), EXTRACTOR)


def convert(pdf: Path):
    """Docling conversion with the same options as the pipeline's extractor."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return converter.convert(str(pdf)).document


def body_items(doc) -> list:
    """Text-bearing items in reading order, body layer only (furniture dropped)."""
    items = []
    for item, _ in doc.iterate_items():
        if not hasattr(item, "text") or not item.text.strip():
            continue
        if str(getattr(item, "content_layer", "")).lower().endswith("furniture"):
            continue
        items.append(item)
    return items


def pick_title(items):
    """The section_header highest on page 1 — provenance, not reading order,
    because Docling emits sample 1's title after the page body."""
    best, best_top = None, -1.0
    for it in items:
        if str(it.label) != "section_header" or not it.prov:
            continue
        prov = it.prov[0]
        if prov.page_no != 1:
            continue
        top = prov.bbox.t if prov.bbox.coord_origin.name == "BOTTOMLEFT" else -prov.bbox.t
        if top > best_top:
            best, best_top = it, top
    return best


def label_items(items, use_cues: bool) -> list[dict]:
    """Map Docling items to stage-2 blocks."""
    title_item = pick_title(items)
    blocks = []
    for it in items:
        text = re.sub(r"[ \t]{2,}", " ", it.text).strip()
        label = str(it.label)
        if it is title_item:
            out = "title"
        elif label == "section_header":
            out = "section.title"
        elif use_cues and len(text) <= QUESTION_MAX_CHARS and QUESTION_CUE.search(text):
            out = "question.text"
        else:
            out = "answer.text"
        blocks.append({"text": text, "label": out, "docling_label": label,
                       "page": it.prov[0].page_no if it.prov else None})
    return blocks


def write_stages(tag: str, n: int, blocks: list[dict]) -> None:
    structured = to_structured(blocks)
    final = apply_new_annotation_rules(json.loads(json.dumps(structured)))
    for path, data in ((P.labeled_path(tag, n), blocks),
                       (P.structured_path(tag, n), structured),
                       (P.final_path(tag, n), final)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.disable(logging.WARNING)

    for n in range(args.start, args.end + 1):
        doc = convert(Path(f"data/input/pdfs/sample{n}.pdf"))
        items = body_items(doc)
        for rule_set, use_cues in RULE_SETS.items():
            blocks = label_items(items, use_cues)
            write_stages(tag_for(rule_set), n, blocks)
            counts = {}
            for b in blocks:
                counts[b["label"]] = counts.get(b["label"], 0) + 1
            print(f"sample{n:<3} {rule_set:11s} {len(blocks):3d} blocks  {counts}")
    print("tags:", ", ".join(tag_for(r) for r in RULE_SETS))


if __name__ == "__main__":
    main()
