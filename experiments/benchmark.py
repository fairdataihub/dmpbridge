"""Benchmark — compare all experiments that have results on disk.

Loads every YAML config in ``experiments/``, evaluates any that already have
output files, and prints a ranked accuracy table.  No model calls are made —
this is a read-only summary of existing results.

Usage
-----
    python experiments/benchmark.py
    python experiments/benchmark.py --dir experiments/
    python experiments/benchmark.py --sort gold_acc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the project root without installing.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpbridge.evaluation.evaluate import gold_metrics
from dmpbridge.evaluation.experiment import EXPERIMENTS_DIR, Experiment, list_experiments


def run_benchmark(
    experiments_dir: Path = EXPERIMENTS_DIR,
    sort_by: str = "block_acc",
) -> list[dict]:
    """Evaluate all experiments that have results on disk.

    Parameters
    ----------
    experiments_dir:
        Directory containing ``*.yaml`` experiment configs.
    sort_by:
        Column to sort by: ``"block_acc"`` | ``"gold_acc"`` | ``"model"``

    Returns
    -------
    list[dict]
        One row per experiment with keys:
        ``name``, ``strategy``, ``model``, ``provider``,
        ``correct``, ``total``, ``block_acc``,
        ``gold_correct``, ``gold_total``, ``gold_acc``, ``tag``.
    """
    rows = []
    for path in list_experiments(experiments_dir):
        exp     = Experiment.from_yaml(path)
        results = exp.evaluate()   # {model: {extractor: (df, conf, errs)}}

        for model, ext_results in results.items():
            for extractor, (df, conf, _) in ext_results.items():
                if df is None or df.empty:
                    continue

                tc = int(df["correct"].sum())
                tn = int(df["total"].sum())

                gold_correct, _, gold_total = gold_metrics(conf)

                rows.append({
                    "name":         exp.config.name,
                    "model":        model,
                    "extractor":    extractor,
                    "strategy":     exp.config.strategy,
                    "provider":     exp.config.provider,
                    "correct":      tc,
                    "total":        tn,
                    "block_acc":    tc / tn if tn else 0.0,
                    "gold_correct": gold_correct,
                    "gold_total":   gold_total,
                    "gold_acc":     gold_correct / gold_total if gold_total else 0.0,
                    "tag":          exp.config.tag_for(model, extractor),
                })

    if sort_by in ("block_acc", "gold_acc", "gold_correct", "correct"):
        rows.sort(key=lambda r: r[sort_by], reverse=True)
    else:
        rows.sort(key=lambda r: (r.get(sort_by, ""), -r["block_acc"]))

    return rows


def print_benchmark(rows: list[dict]) -> None:
    """Print the benchmark table to stdout."""
    if not rows:
        print("No results found. Run experiments first with:  dmpbridge-experiment <config.yaml>")
        return

    header = (
        f"  {'Model':<22}  {'Extractor':<12}  {'Blocks':>8}"
        f"  {'Block acc':>10}  {'Gold acc':>10}  Tag"
    )
    print(header)
    print("  " + "-" * (len(header) + 20))

    for r in rows:
        print(
            f"  {r['model']:<22}  {r['extractor']:<12}  {r['total']:>8}"
            f"  {r['block_acc']*100:>9.1f}%  {r['gold_acc']*100:>9.1f}%  {r['tag']}"
        )

    print()
    best_block = max(rows, key=lambda r: r["block_acc"])
    best_gold  = max(rows, key=lambda r: r["gold_acc"])
    print(f"  Best block acc : {best_block['name']}  ({best_block['block_acc']*100:.1f}%)")
    print(f"  Best gold acc  : {best_gold['name']}  ({best_gold['gold_acc']*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print an accuracy comparison table for all evaluated experiments."
    )
    ap.add_argument(
        "--dir", default=str(EXPERIMENTS_DIR), type=Path,
        help="Directory containing experiment YAML files (default: %(default)s)",
    )
    ap.add_argument(
        "--sort", default="block_acc",
        choices=["block_acc", "gold_acc", "model", "strategy"],
        help="Column to sort by (default: %(default)s)",
    )
    ap.add_argument(
        "--pandas", action="store_true",
        help="Print using pandas instead of plain text (requires pandas).",
    )
    args = ap.parse_args()

    rows = run_benchmark(experiments_dir=args.dir, sort_by=args.sort)

    if args.pandas:
        import pandas as pd
        df = pd.DataFrame(rows)
        df["block_acc"] = df["block_acc"].map("{:.1%}".format)
        df["gold_acc"]  = df["gold_acc"].map("{:.1%}".format)
        print(df[["name", "strategy", "model", "total", "block_acc", "gold_acc"]].to_string(index=False))
    else:
        print_benchmark(rows)


if __name__ == "__main__":
    main()
