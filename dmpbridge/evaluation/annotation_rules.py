"""Convert structured DMP JSON to the new-annotation style, and evaluate it.

This is "Path B" alongside the main ``dmpbridge-evaluate`` command ("Path A"):

    Path A (evaluate.py)      structured JSON  ->  scored against OLD annotation
    Path B (this module)      structured JSON  ->  rule-converted "final JSON"
                                                 ->  scored against NEW annotation

The rules are specified by ``data/input/Rules.xlsx`` — a 16-row truth table over
whether each of ``title``, ``question.text``, ``section.title`` and
``section.description`` is empty (E) or non-empty (N).  Exactly one row matches
any given question, and ``_RULES`` below is a direct transcription of the sheet:
the spreadsheet is the specification, and this module executes it literally
rather than interpreting it.

Six rows instruct a copy that cannot succeed as a fill, and are executed anyway
because the sheet is authoritative:

    rows 3, 11   source (section.description) is marked empty, so the copy
                 writes an empty string — effectively a no-op
    rows 5, 13   source (section.title) is marked empty AND the target
                 (question.text) is marked non-empty, so the copy CLEARS
                 existing question text
    rows 7, 15   target (question.text) is marked non-empty, so the copy
                 OVERWRITES it with the section title

Note also that every row carrying an action has ``section.description`` empty,
and every row with a description says "leave unchanged" — so a section that has
a description is never modified.

Unlike earlier revisions, no row writes into ``section.title``; ``question.text``
is the only field this module ever assigns.

A pattern seen in the ground truth but NOT covered by the table — merging several
same-section sub-questions (e.g. "Raw data:", "Scripts and code for analyses:")
into one with their answers concatenated — remains unimplemented.  It is real
(confirmed on sample5) but not mechanically derivable from the JSON alone:
sample1 and sample2 have superficially identical short, unlettered sub-question
labels that were *not* merged, so there is no reliable trigger condition to
encode without guessing.  See notebooks/annotation_conversion_test.ipynb.

Usage (CLI):
    dmpbridge-evaluate-new                                        # all tags
    dmpbridge-evaluate-new llama3.1-8b_pdfplumber_whole_doc        # one tag
    dmpbridge-evaluate-new --convert-only                          # build final JSON only, skip scoring
"""
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from ..utils import get_logger, setup_logging
from .evaluate import (
    LABELS,
    LLM_DIR,
    NEW_MANUAL_DIR,
    _confusion_from_match,
    _match_structured,
    _snum,
    add_confusion,
    compute_f1_rows,
    confusion_matrix_df,
    extract_gold,
    gold_metrics,
    list_tags,
    micro_prf1,
    print_confusion,
    print_f1,
    print_micro_prf1,
    print_missed,
)

logger = get_logger(__name__)

from ..core import paths as _paths

FINAL_DIR = _paths.FINAL_DIR   # stage 4 — rule-converted output


# ── The rule table ────────────────────────────────────────────────────────────

# Direct transcription of data/input/Rules.xlsx.  Key is the emptiness pattern
# (title, question.text, section.title, section.description); value is
# (source_field, target_field) or None for "leave it unchanged".
#
# Keep this aligned with the spreadsheet row for row.  Do not "fix" a row here —
# if a row looks wrong, correct the sheet and re-transcribe, so the two cannot
# drift apart.
_RULES: dict[tuple[str, str, str, str], tuple[str, str] | None] = {
    # row  title q.text sec.title sec.desc      action
    ("E", "E", "E", "E"): None,                                        # 1
    ("E", "E", "E", "N"): None,                                        # 2
    ("E", "E", "N", "E"): ("section.description", "question.text"),    # 3
    ("E", "E", "N", "N"): None,                                        # 4
    ("E", "N", "E", "E"): ("section.title", "question.text"),          # 5
    ("E", "N", "E", "N"): None,                                        # 6
    ("E", "N", "N", "E"): ("section.title", "question.text"),          # 7
    ("E", "N", "N", "N"): None,                                        # 8
    ("N", "E", "E", "E"): ("title", "question.text"),                  # 9
    ("N", "E", "E", "N"): None,                                        # 10
    ("N", "E", "N", "E"): ("section.description", "question.text"),    # 11
    ("N", "E", "N", "N"): None,                                        # 12
    ("N", "N", "E", "E"): ("section.title", "question.text"),          # 13
    ("N", "N", "E", "N"): None,                                        # 14
    ("N", "N", "N", "E"): ("section.title", "question.text"),          # 15
    ("N", "N", "N", "N"): None,                                        # 16
}


def _state(value: str) -> str:
    """'N' when *value* holds text, 'E' otherwise."""
    return "N" if (value or "").strip() else "E"


def apply_new_annotation_rules(data: dict) -> dict:
    """Return a copy of *data* with the Rules.xlsx truth table applied.

    One row of ``_RULES`` matches each question; its action is carried out
    exactly as the spreadsheet states it, including the six rows whose copy
    empties or overwrites ``question.text`` (see the module docstring).  The
    spreadsheet is the specification — this function does not second-guess it.
    """
    data     = copy.deepcopy(data)
    root     = data.get("narrative", data)
    template = root.get("template", {})
    doc_title = (template.get("title") or "").strip()

    for section in template.get("section", []):
        for question in section.get("question", []):
            values = {
                "title":               doc_title,
                "question.text":       (question.get("text") or "").strip(),
                "section.title":       (section.get("title") or "").strip(),
                "section.description": (section.get("description") or "").strip(),
            }
            action = _RULES[tuple(_state(values[f]) for f in (
                "title", "question.text", "section.title", "section.description"))]
            if action is None:
                continue

            source, target = action
            if target == "question.text":
                question["text"] = values[source]
            elif target == "section.title":
                section["title"] = values[source]
            elif target == "section.description":
                section["description"] = values[source]
            else:                                   # pragma: no cover
                raise ValueError(f"Rules.xlsx: unknown target field {target!r}")

    return data


def resolve_new_gt_path(n: int) -> Path:
    """Resolve sample n's new-version ground truth file, regardless of naming pattern.

    Filenames aren't consistent: samples 1,2,3,4,7 use sampleN_dmp_new.json;
    samples 5,6,8,9,10 use dmp_sampleN_new.json. This scans by sample number
    instead of hardcoding either pattern.
    """
    for p in NEW_MANUAL_DIR.glob("*.json"):
        m = re.fullmatch(r"\D*(\d+)\D*", p.stem)
        if m and int(m.group(1)) == n:
            return p
    raise FileNotFoundError(f"No new-version ground truth file found for sample{n} in {NEW_MANUAL_DIR}")


# ── Step 1: build final JSON ─────────────────────────────────────────────────

def convert_tag_to_final(tag: str) -> int:
    """Apply the annotation rule to every structured JSON prediction under *tag*.

    Reads stage 3 and writes stage 4, mirroring the per-tag layout.
    Returns the number of files converted.
    """
    src_dir = _paths.STRUCTURED_DIR / tag
    dst_dir = FINAL_DIR / tag
    files   = sorted(src_dir.glob("sample*.json"))
    if files:
        dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        pred  = json.loads(src.read_text(encoding="utf-8"))
        final = apply_new_annotation_rules(pred)
        (dst_dir / src.name).write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    logger.info("%s: converted %d file(s) -> %s", tag, len(files), dst_dir)
    return len(files)


# ── Step 2: evaluate final JSON against the new-version ground truth ────────

def load_method_new(tag: str, exclude: list[int] | None = None):
    """Evaluate every final-JSON sample for *tag* against the new-version ground truth.

    Mirrors ``evaluate.load_method()``, but reads FINAL_DIR predictions and
    resolves gold via ``resolve_new_gt_path()`` instead of the fixed
    old-annotation pair.

    Returns
    -------
    tuple[pd.DataFrame, dict, pd.DataFrame] | tuple[None, None, None]
    """
    import pandas as pd

    _exclude = set(exclude or [])
    tag_dir  = FINAL_DIR / tag
    if not tag_dir.exists():
        return None, None, None

    files = sorted(tag_dir.glob("sample*.json"), key=_snum)
    found = [f for f in files if _snum(f) not in _exclude]
    if not found:
        return None, None, None

    conf_all: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows, errors = [], []

    for f in files:
        n = _snum(f)
        if n in _exclude:
            continue
        stem             = f"sample{n}"
        gold             = extract_gold(resolve_new_gt_path(n))
        records, no_gold = _match_structured(f, gold)
        conf             = _confusion_from_match(records, no_gold)
        add_confusion(conf_all, conf)

        for r in records:
            if r["pred_label"] is not None and r["pred_label"] != r["gold_label"]:
                errors.append({
                    "sample": stem,
                    "text":   r["pred_text"][:120],
                    "true":   r["gold_label"],
                    "pred":   r["pred_label"],
                })
        for pred_text, pred_label in no_gold:
            errors.append({
                "sample": stem,
                "text":   pred_text[:120],
                "true":   "no_gold_match",
                "pred":   pred_label,
            })

        correct, _, total = gold_metrics(conf)
        rows.append({
            "sample":   stem,
            "total":    total,
            "correct":  correct,
            "errors":   total - correct,
            "accuracy": correct / total if total else 0,
            "formula":  f"{correct}/{total}",
        })

    return pd.DataFrame(rows), conf_all, pd.DataFrame(errors)


def run_single(final_path: Path) -> None:
    """Evaluate one final-JSON file against its matching new-version ground truth."""
    n = _snum(final_path)
    try:
        gt_path = resolve_new_gt_path(n)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return

    gold             = extract_gold(gt_path)
    records, no_gold = _match_structured(final_path, gold)
    confusion        = _confusion_from_match(records, no_gold)
    tp = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
    n_total = sum(sum(v.values()) for v in confusion.values())

    print("\n" + "=" * 62)
    print(f"  {final_path.name}  vs  {gt_path.name}  (new annotation)")
    print("=" * 62 + "\n")
    print(f"{'Sample':<12}  {'Total':>5}  {'Correct':>7}  {'Errors':>6}  {'Accuracy':>8}  Formula")
    print("-" * 58)
    errors_n = n_total - tp
    acc      = tp / n_total * 100 if n_total else 0.0
    print(f"sample{n:<6}  {n_total:>5}  {tp:>7}  {errors_n:>6}  {acc:>7.1f}%  {tp}/{n_total}")
    print("\nConfusion matrix  (* = correct, ! = missed)\n")
    print_confusion(confusion)
    print_f1(confusion)
    print()
    print_micro_prf1(confusion)
    print("\nMissed gold items (final JSON produced no matching block):")
    print_missed(records)


def run_all(tag: str, exclude: list[int] | None = None) -> None:
    """Build final JSON for *tag*, then evaluate it against the new-version ground truth."""
    _exclude = set(exclude or [])
    n_converted = convert_tag_to_final(tag)
    if n_converted == 0:
        print(f"No structured JSON found for tag {tag!r} under {_paths.STRUCTURED_DIR / tag}")
        return

    df, conf, _errors = load_method_new(tag, exclude=_exclude)
    if df is None:
        print(f"No final JSON found for tag {tag!r} under {FINAL_DIR / tag}")
        return

    excl_note = f"  (excluding sample{sorted(_exclude)} — used for prompt development)" if _exclude else ""
    print(f"\nEvaluating (new annotation): {tag}{excl_note}\n")
    print(f"{'Sample':<12}  {'Total':>5}  {'Correct':>7}  {'Errors':>6}  {'Accuracy':>8}  Formula")
    print("-" * 58)
    for row in df.itertuples():
        print(f"{row.sample:<12}  {row.total:>5}  {row.correct:>7}  {row.errors:>6}  "
              f"{row.accuracy*100:>7.1f}%  {row.formula}")
    print("-" * 58)
    tc, tn = int(df["correct"].sum()), int(df["total"].sum())
    tp_all = sum(conf.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
    n_all  = sum(sum(v.values()) for v in conf.values())
    print(f"{'TOTAL':<12}  {n_all:>5}  {tp_all:>7}  {n_all-tp_all:>6}  "
          f"{tp_all/n_all*100 if n_all else 0:>7.1f}%  {tp_all}/{n_all}")

    print("\n" + "=" * 62)
    print("  AGGREGATE  (all samples pooled, new annotation)")
    print("=" * 62)
    print("\nConfusion matrix  (* = correct, ! = missed)\n")
    print_confusion(conf)
    print_f1(conf)
    print()
    print_micro_prf1(conf)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: dmpbridge-evaluate-new."""
    setup_logging()

    import argparse
    ap = argparse.ArgumentParser(
        description="Convert structured DMP JSON to the new-annotation style and "
                     "evaluate it against data/input/ground_truth_new_version.",
    )
    ap.add_argument(
        "target", nargs="?",
        help="Result tag (e.g. llama3.1-8b_pdfplumber_whole_doc) or path to a single "
             "final '*_structured.json' file (already converted, under labeled_final/).",
    )
    ap.add_argument("--list", "-l", action="store_true", help="List available result tags and exit.")
    ap.add_argument("--exclude", "-x", default="",
                    help="Comma-separated sample numbers to skip, e.g. --exclude 1,2")
    ap.add_argument("--convert-only", action="store_true",
                    help="Only build final JSON (Step 1) — skip scoring against the new annotation.")
    args = ap.parse_args()
    exclude = [int(n) for n in args.exclude.split(",") if n.strip().isdigit()]

    if args.list:
        tags = list_tags()
        if not tags:
            print(f"No results found in {_paths.STRUCTURED_DIR}")
        else:
            print(f"Available tags in {LLM_DIR}:\n")
            for t in tags:
                n = len(list((_paths.STRUCTURED_DIR / t).glob("sample*.json")))
                print(f"  {t}  ({n} samples)")
        return

    if args.target and Path(args.target).is_file():
        run_single(Path(args.target))
        return

    if args.convert_only:
        tags = [args.target] if args.target else list_tags()
        for tag in tags:
            convert_tag_to_final(tag)
        return

    if args.target:
        run_all(args.target, exclude=exclude)
        return

    # No argument — auto-detect: use the only tag or prompt the user.
    tags = list_tags()
    if not tags:
        print(f"No results found in {_paths.STRUCTURED_DIR}. Run an experiment first.")
        sys.exit(1)
    if len(tags) == 1:
        run_all(tags[0], exclude=exclude)
    else:
        print("Multiple result sets found. Specify one with:\n")
        for t in tags:
            print(f"  dmpbridge-evaluate-new {t}")
        sys.exit(1)


if __name__ == "__main__":
    main()
