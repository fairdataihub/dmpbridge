"""Run one configuration N times and report the spread.

Establishes what counts as a real difference. Measured 2026-08-06 on
llama3.1:8b + pdfplumber: three runs, every count identical, F1 spread 0.000.
Re-measure whenever the model, extractor or Ollama version changes — the answer
is a property of the stack, not a constant.

    python scripts/noise_floor.py
    python scripts/noise_floor.py --model gemma4:e4b --runs 3

Each run clears stages 2-4 and re-labels; stage 1 extraction is reused, so this
measures the model, not the extractor.
"""
import argparse
import statistics
import subprocess
import sys

from dmpbridge.evaluation.annotation_rules import load_method_new
from dmpbridge.evaluation.evaluate import LABELS, compute_f1_rows, load_method, micro_prf1
from rerun import clear, tag_for   # same directory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--extractor", default="pdfplumber",
                    choices=["pdfplumber"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    args = ap.parse_args()

    tag = tag_for(args.model, args.extractor)
    print(f"{args.model} + {args.extractor}, {args.runs} runs, samples "
          f"{args.start}-{args.end}\n")

    rows = []
    for i in range(1, args.runs + 1):
        clear(tag)
        r = subprocess.run(
            ["dmpbridge-wholedoc", "--model", args.model, "--extractor", args.extractor,
             "--start", str(args.start), "--end", str(args.end)],
            capture_output=True, text=True, check=False)
        if r.returncode != 0:
            print(f"run {i} FAILED (exit {r.returncode})")
            print(r.stderr[-800:])
            sys.exit(r.returncode)

        _, conf_a, _ = load_method(tag, exclude=[])
        conf_b = load_method_new(tag, exclude=[])[1]
        rows.append((micro_prf1(conf_a),
                     micro_prf1(conf_b) if conf_b is not None else None,
                     compute_f1_rows(conf_a).set_index("label")))
        print(f"run {i}: f1 {rows[-1][0]['f1']:.3f}")

    hdr = f"\n{'':<12}" + "".join(f"{'run ' + str(i):>9}" for i in range(1, args.runs + 1))
    print(hdr + f"{'spread':>10}")
    print("-" * (len(hdr) + 9))
    for k in ("tp", "fp", "fn"):
        v = [r[0][k] for r in rows]
        print(f"  {k.upper():<10}" + "".join(f"{x:>9}" for x in v) + f"{max(v)-min(v):>10}")
    for k in ("precision", "recall", "f1"):
        v = [r[0][k] for r in rows]
        print(f"  {k:<10}" + "".join(f"{x:>9.3f}" for x in v) + f"{max(v)-min(v):>10.3f}")

    print("\nPer-class f1 spread (Path A):")
    for lab in LABELS:
        v = [r[2].loc[lab, "f1"] for r in rows]
        print(f"  {lab:<22}{max(v)-min(v):>8.3f}")

    fa = [r[0]["f1"] for r in rows]
    spread = max(fa) - min(fa)
    print(f"\nPath A f1: mean {statistics.mean(fa):.4f}, spread {spread:.4f}")
    if rows[0][1]:
        fb = [r[1]["f1"] for r in rows]
        print(f"Path B f1: mean {statistics.mean(fb):.4f}, spread {max(fb)-min(fb):.4f}")

    print(f"\nTreat differences at or below {max(spread, 0.002):.3f} F1 as "
          f"indistinguishable from no change.")
    print("Update NOISE_FLOOR in scripts/compare_results.py and the figure in "
          "CLAUDE.md if this differs from what they record.")


if __name__ == "__main__":
    main()
