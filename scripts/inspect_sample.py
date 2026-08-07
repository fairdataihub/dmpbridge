"""Follow one document through all four stages, and score just that document.

The whole-document runner works in batches; this is for looking closely at a
single sample — what the extractor produced, what the model called each block,
what structure came out, and exactly which blocks were scored wrong.

    python scripts/inspect_sample.py 1
    python scripts/inspect_sample.py 1 --model gemma4:e4b
    python scripts/inspect_sample.py 1 --rerun          # re-label this sample only
    python scripts/inspect_sample.py 1 --blocks         # every block, not a summary
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from dmpbridge.core import paths as P
from dmpbridge.evaluation.evaluate import (
    _confusion_from_match, _match_structured, extract_gold, micro_prf1,
    resolve_old_gt_path,
)

STAGES = [("2_labeled", P.LABELED_DIR), ("3_structured", P.STRUCTURED_DIR),
          ("4_final", P.FINAL_DIR)]


def tag_for(model: str, extractor: str) -> str:
    return f"{model.replace(':', '-')}_{extractor}_whole_doc"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def blocks_of(data) -> list:
    """Stage 1 and 2 are either a bare list or {'blocks': [...]}."""
    if data is None:
        return []
    return data if isinstance(data, list) else data.get("blocks", [])


def walk_structured(data) -> tuple[str, list]:
    """Return (title, sections) from the DMP-tool narrative schema."""
    tpl = (data or {}).get("narrative", {}).get("template", {})
    return tpl.get("title", ""), tpl.get("section", [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", type=int)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--extractor", default="pdfplumber",
                    choices=["pdfplumber", "docling", "lighton"])
    ap.add_argument("--rerun", action="store_true",
                    help="clear and re-label this sample before inspecting")
    ap.add_argument("--blocks", action="store_true", help="list every block")
    args = ap.parse_args()

    n, tag = args.sample, tag_for(args.model, args.extractor)
    print(f"sample{n}  ·  {args.model}  ·  {args.extractor}\n")

    if args.rerun:
        # Delete only this sample's files — the rest of the tag stays cached.
        for name, root in STAGES:
            (root / tag / f"sample{n}.json").unlink(missing_ok=True)
        subprocess.run(["dmpbridge-wholedoc", "--model", args.model,
                        "--extractor", args.extractor,
                        "--start", str(n), "--end", str(n)], check=True)
        print()

    s1 = blocks_of(load(P.EXTRACTED_DIR / args.extractor / f"sample{n}.json"))
    s2 = blocks_of(load(P.LABELED_DIR / tag / f"sample{n}.json"))
    title, sections = walk_structured(load(P.STRUCTURED_DIR / tag / f"sample{n}.json"))
    _, final_secs = walk_structured(load(P.FINAL_DIR / tag / f"sample{n}.json"))

    counts: dict[str, int] = {}
    for b in s2:
        counts[b.get("label", "?")] = counts.get(b.get("label", "?"), 0) + 1

    print(f"stage 1  extracted    {len(s1)} blocks")
    print(f"stage 2  labeled      {len(s2)} blocks  ->  "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    nq = sum(len(s.get("question", [])) for s in sections)
    print(f"stage 3  structured   title {title[:40]!r}, {len(sections)} sections, {nq} questions")
    nqf = sum(len(s.get("question", [])) for s in final_secs)
    print(f"stage 4  final        {len(final_secs)} sections, {nqf} questions "
          f"(rules filled {nqf - nq:+d})")

    if args.blocks:
        print("\nEvery block, as labeled:")
        for i, b in enumerate(s2):
            print(f"  {i:>3} [{b.get('label', '?'):<20}] {b.get('text', '')[:70]!r}")

    # ── Score this one document ──────────────────────────────────────────
    pred = P.STRUCTURED_DIR / tag / f"sample{n}.json"
    if not pred.exists():
        print("\nNo stage 3 output — nothing to score.")
        return

    gold = extract_gold(resolve_old_gt_path(n))
    records, no_gold = _match_structured(pred, gold)
    conf = _confusion_from_match(records, no_gold)
    m = micro_prf1(conf)

    print(f"\nPath A score for this document alone")
    print(f"  TP {m['tp']}   FP {m['fp']}   FN {m['fn']}")
    print(f"  precision {m['precision']:.3f}   recall {m['recall']:.3f}   f1 {m['f1']:.3f}")

    wrong = [r for r in records if r["pred_label"] and r["pred_label"] != r["gold_label"]]
    missed = [r for r in records if r["pred_label"] is None]
    if wrong:
        print(f"\n  Wrong label ({len(wrong)}):")
        for r in wrong:
            print(f"    {r['gold_label']} -> {r['pred_label']}")
            print(f"        {r['pred_text'][:66]!r}")
    if missed:
        print(f"\n  In the annotation, not produced ({len(missed)}):")
        for r in missed:
            print(f"    [{r['gold_label']}] {str(r.get('gold_text', ''))[:60]!r}")
    if no_gold:
        print(f"\n  Produced, not in the annotation ({len(no_gold)}):")
        for text, label in no_gold[:10]:
            print(f"    [{label}] {text[:60]!r}")
        if len(no_gold) > 10:
            print(f"    ... and {len(no_gold) - 10} more")


if __name__ == "__main__":
    main()
