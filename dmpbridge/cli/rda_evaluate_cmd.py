"""Score the pipeline's RDA JSON (stage 5) against the hand-made references.

The same rules as the RDA notebooks (see dmpbridge.rda.evaluate). For each sample in
data/output/5_rda/ with a reference in data/input/rda_ground_truth/:

* the field matrix - every annotated field Correct, Mismatch or Missed
* every field - Correct, Hallucinated or Missed, with precision, recall and F1

Usage:
    dmpbridge-evaluate-rda                      every sample in 5_rda/
    dmpbridge-evaluate-rda --start 3 --end 3

Writes, in 5_rda/evaluation/:
    matrix_sampleN.png          the field matrix for one sample
    evaluation_results.xlsx     summary, per-sample scores, field matrix, every field
"""
import argparse
import re
from pathlib import Path
from typing import Optional

from ..core import paths
from ..rda import evaluate as E
from ..rda.stages import read_run_info


def evaluate_rda(samples: Optional[list[int]] = None, reference_dir: Path = E.REFERENCE_DIR,
                 save: bool = True) -> Optional[dict]:
    """Score every sample of 5_rda/ that has a reference; print the results and, with
    *save*, write the matrix figures and the spreadsheet. Returns the pooled metrics,
    or None when there was nothing to score."""
    found = sorted(int(m.group(1)) for p in paths.RDA_DIR.glob("sample*.json")
                   if (m := re.fullmatch(r"sample(\d+)\.json", p.name)))
    if samples is not None:
        found = [n for n in found if n in samples]
    scored = [n for n in found if E.reference_path(n, reference_dir).exists()]
    for n in sorted(set(found) - set(scored)):
        print(f"sample{n}: no reference in {reference_dir} - not scored")
    if not scored:
        print(f"nothing to score in {paths.RDA_DIR}")
        return None

    def model_of(n):
        return (read_run_info(paths.rda_path(n)) or {}).get("rda_model", "unknown model")

    out_dir = paths.RDA_DIR / "evaluation"
    all_rows, n_ref_total, per_sample, matrix_rows, detail_rows = [], 0, [], [], []

    print(f"\n{len(scored)} sample(s) in {paths.RDA_DIR} with a reference\n")
    print(f"{'sample':8} {'RDA model':14} {'annotated':>9} {'correct':>8} {'mismatch':>9} {'missed':>7}"
          f"   {'precision':>9} {'recall':>7} {'F1':>6}")
    print("-" * 89)
    for n in scored:
        model = model_of(n)
        rows, m, outcomes = E.evaluate_file(paths.rda_path(n), E.reference_path(n, reference_dir))
        counts = {k: sum(v == k for v in outcomes.values()) for k in E.OUTCOMES}
        print(f"sample{n:<2} {model:14} {len(outcomes):>9} {counts['Correct']:>8} {counts['Mismatch']:>9} "
              f"{counts['Missed']:>7}   {m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>6.3f}")
        all_rows += rows
        n_ref_total += m["fields in reference"]
        per_sample.append({"sample": n, "rda model": model, "annotated fields": len(outcomes),
                           **{k.lower(): v for k, v in counts.items()}, **m})
        matrix_rows += [{"sample": n, "rda model": model, "field": f, "part of the form": E.row_of(f)[1],
                         "outcome": o} for f, o in sorted(outcomes.items())]
        detail_rows += [{"sample": n, "rda model": model, **r} for r in rows]
        if save:
            E.matrix_figure({model: outcomes},
                            f"What {model} did with the {len(outcomes)} annotated fields - sample {n}",
                            out_dir / f"matrix_sample{n}.png")
    pooled = E.metrics(all_rows, n_ref_total)
    if len(scored) > 1:
        print("-" * 89)
        totals = {k: sum(r["outcome"] == k for r in matrix_rows) for k in E.OUTCOMES}
        print(f"{'all':8} {'':14} {len(matrix_rows):>9} {totals['Correct']:>8} {totals['Mismatch']:>9} "
              f"{totals['Missed']:>7}   {pooled['precision']:>9.3f} {pooled['recall']:>7.3f} "
              f"{pooled['f1']:>6.3f}")
    print("\nannotated fields: Correct / Mismatch / Missed (the field matrix)")
    print("precision, recall, F1: over every field either side wrote (Correct / Hallucinated / Missed)")

    if save:
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.close("all")
        out_dir.mkdir(parents=True, exist_ok=True)
        models = ", ".join(sorted({r["rda model"] for r in per_sample}))
        with pd.ExcelWriter(out_dir / "evaluation_results.xlsx") as w:
            pd.DataFrame([{"rda model": models, "samples": len(scored), **pooled}]).to_excel(
                w, sheet_name="summary", index=False)
            pd.DataFrame(per_sample).to_excel(w, sheet_name="per sample", index=False)
            pd.DataFrame(matrix_rows).to_excel(w, sheet_name="field matrix", index=False)
            pd.DataFrame(detail_rows).astype({"model value": str, "reference value": str}).to_excel(
                w, sheet_name="every field", index=False)
        print(f"\nsaved -> {out_dir}  (matrix_sampleN.png, evaluation_results.xlsx)")
    return pooled


def main() -> None:
    """CLI entry point: dmpbridge-evaluate-rda."""
    ap = argparse.ArgumentParser(description="Score the pipeline's RDA JSON against the hand-made references.")
    ap.add_argument("--start", type=int, default=None, help="first sample (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last sample (inclusive)")
    ap.add_argument("--no-save", action="store_true", help="print only; write no figure or spreadsheet")
    args = ap.parse_args()
    samples = None
    if args.start is not None or args.end is not None:
        samples = list(range(args.start or 1, (args.end or 10_000) + 1))
    evaluate_rda(samples, save=not args.no_save)


if __name__ == "__main__":
    main()
