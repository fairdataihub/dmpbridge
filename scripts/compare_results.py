"""Score every completed configuration and print a comparison, both paths.

This exists because the same ad-hoc scoring snippet kept being rewritten by hand,
which is how a table ends up quoting one path for one model and the other path
for another.

    python scripts/compare_results.py                    # all tags found on disk
    python scripts/compare_results.py --tag gemma4-e4b_pdfplumber_whole_doc
    python scripts/compare_results.py --per-class        # add the per-label table
    python scripts/compare_results.py --questions        # where question.text went
    python scripts/compare_results.py --baseline old.json --save new.json

`--save` writes the scores to JSON; passing that file back as `--baseline` on a
later run prints a change column and flags anything outside the noise floor.
"""
import argparse
import json
from pathlib import Path

from dmpbridge.core import paths as P
from dmpbridge.evaluation.annotation_rules import load_method_new
from dmpbridge.evaluation.evaluate import (
    LABELS, compute_f1_rows, load_method, micro_prf1,
)

# Measured 2026-08-06 over four identical runs: every count identical bar one
# block. Differences at or below this are not interpretable.
NOISE_FLOOR = 0.005


def discover_tags() -> list[str]:
    """Every tag with stage 3 output, newest first."""
    if not P.STRUCTURED_DIR.exists():
        return []
    return sorted((d.name for d in P.STRUCTURED_DIR.iterdir() if d.is_dir()),
                  key=lambda t: -(P.STRUCTURED_DIR / t).stat().st_mtime)


def score(tag: str) -> dict | None:
    """Micro scores for both paths, plus per-class and the question.text row."""
    df_a, conf_a, _ = load_method(tag, exclude=[])
    if df_a is None:
        return None
    res_b = load_method_new(tag, exclude=[])
    conf_b = res_b[1]

    out = {
        "tag": tag,
        "documents": len(df_a),
        "path_a": micro_prf1(conf_a),
        "per_class": {lab: dict(r) for lab, r in
                      compute_f1_rows(conf_a).set_index("label").iterrows()},
        "questions": {k: v for k, v in conf_a.get("question.text", {}).items() if v},
    }
    out["path_b"] = micro_prf1(conf_b) if conf_b is not None else None
    return out


def delta(new: float, old: float | None) -> str:
    """Change column, blanked when the move is inside the noise floor."""
    if old is None:
        return ""
    d = new - old
    if abs(d) < NOISE_FLOOR:
        return f"{d:+.3f} ~"       # ~ marks "indistinguishable from no change"
    return f"{d:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", help="score only this tag (repeatable)")
    ap.add_argument("--per-class", action="store_true", help="add the per-label table")
    ap.add_argument("--questions", action="store_true",
                    help="show where each model sent the real question.text blocks")
    ap.add_argument("--baseline", type=Path, help="a previous --save file to compare against")
    ap.add_argument("--save", type=Path, help="write these scores to JSON")
    args = ap.parse_args()

    tags = args.tag or discover_tags()
    if not tags:
        print("No scored configurations found under", P.STRUCTURED_DIR)
        return

    base = {}
    if args.baseline and args.baseline.exists():
        base = {r["tag"]: r for r in json.loads(args.baseline.read_text(encoding="utf-8"))}

    results = [r for r in (score(t) for t in tags) if r]
    if not results:
        print("No tag had scoreable output.")
        return

    w = max(len(r["tag"]) for r in results) + 2
    head = f"{'configuration':<{w}}{'docs':>5}{'prec':>8}{'recall':>8}{'Path A':>9}{'Path B':>9}"
    if base:
        head += f"{'A change':>11}"
    print(head)
    print("-" * len(head))

    for r in results:
        a, b = r["path_a"], r["path_b"]
        line = (f"{r['tag']:<{w}}{r['documents']:>5}{a['precision']:>8.3f}"
                f"{a['recall']:>8.3f}{a['f1']:>9.3f}"
                f"{(b['f1'] if b else float('nan')):>9.3f}")
        if base:
            prev = base.get(r["tag"])
            line += f"{delta(a['f1'], prev['path_a']['f1'] if prev else None):>11}"
        print(line)

    if base:
        print(f"\n  ~ marks a change within the +/-{NOISE_FLOOR} noise floor "
              f"— not interpretable as an improvement or regression.")

    if args.per_class:
        for r in results:
            print(f"\n{r['tag']} — per class (Path A)")
            print(f"  {'':<22}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
            for lab in LABELS:
                m = r["per_class"].get(lab)
                if m:
                    print(f"  {lab:<22}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                          f"{m['f1']:>8.3f}{int(m['support']):>9}")

    if args.questions:
        print("\nWhere the real question.text blocks ended up (Path A):")
        for r in results:
            row = ", ".join(f"{v} -> {'nothing' if k == '__missed__' else k}"
                            for k, v in sorted(r["questions"].items(), key=lambda x: -x[1]))
            print(f"  {r['tag']:<{w}}{row}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
