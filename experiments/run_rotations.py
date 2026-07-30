"""Run all 5 leave-2-out rotations and report mean ± std gold_acc.

Usage
-----
    python experiments/run_rotations.py                     # run + evaluate all 5
    python experiments/run_rotations.py --evaluate-only     # skip inference, report existing results
    python experiments/run_rotations.py --dir experiments/rotations

The script discovers every YAML in the rotations directory, runs each experiment,
collects per-sample gold_acc, and prints a summary table with mean ± std.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpbridge.evaluation.experiment import Experiment
from dmpbridge.evaluation.evaluate import extract_gold, evaluate_sample, gold_metrics, MANUAL_DIR

ROTATIONS_DIR = Path(__file__).parent / "rotations"


def _gold_acc(exp: Experiment) -> dict[int, float]:
    """Return {sample_id: gold_acc} for every evaluated sample in this experiment."""
    cfg      = exp.config
    few_shot = set(cfg.few_shot_samples)
    eval_ids = set(cfg.eval_samples) if cfg.eval_samples else (set(cfg.sample_range) - few_shot)

    out_dir = Path(cfg.out_dir) / cfg.tag
    accs: dict[int, float] = {}

    for sid in sorted(eval_ids):
        pred_path = out_dir / f"sample{sid}.json"
        gold_path = MANUAL_DIR / f"sample{sid}_dmp.json"
        if not pred_path.exists() or not gold_path.exists():
            continue
        gold_pairs       = extract_gold(gold_path)
        confusion        = evaluate_sample(pred_path, gold_pairs)
        correct, _, total = gold_metrics(confusion)
        if total > 0:
            accs[sid] = correct / total
    return accs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all leave-2-out rotations and report results.")
    ap.add_argument(
        "--dir",
        default=str(ROTATIONS_DIR),
        help="Directory containing rotation YAML configs (default: experiments/rotations).",
    )
    ap.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip inference — only report accuracy for existing result files.",
    )
    args = ap.parse_args()

    rotation_dir = Path(args.dir)
    yaml_files   = sorted(rotation_dir.glob("*.yaml"))

    if not yaml_files:
        print(f"No YAML files found in {rotation_dir}")
        sys.exit(1)

    print("\nLeave-2-out rotation evaluation")
    print(f"Configs : {rotation_dir}")
    print(f"Rotations: {len(yaml_files)}\n")
    print(f"{'Rotation':<55}  {'Few-shot':>10}  {'N eval':>6}  {'Gold acc':>9}")
    print("-" * 90)

    all_accs: list[float] = []
    per_sample: dict[int, list[float]] = {}

    for yaml_path in yaml_files:
        exp = Experiment.from_yaml(yaml_path)
        cfg = exp.config

        if not args.evaluate_only:
            exp.run()

        accs = _gold_acc(exp)
        if not accs:
            print(f"  {cfg.name:<53}  — no results found")
            continue

        rotation_mean = sum(accs.values()) / len(accs)
        all_accs.extend(accs.values())
        for sid, acc in accs.items():
            per_sample.setdefault(sid, []).append(acc)

        few_str = str(cfg.few_shot_samples)
        print(
            f"  {cfg.name:<53}  {few_str:>10}  {len(accs):>6}  {rotation_mean:>8.1%}"
        )

    if not all_accs:
        print("\nNo results to summarise — run without --evaluate-only first.")
        return

    overall_mean = statistics.mean(all_accs)
    overall_std  = statistics.stdev(all_accs) if len(all_accs) > 1 else 0.0

    print("-" * 90)
    print(f"\n  Overall mean gold_acc : {overall_mean:.1%}")
    print(f"  Std dev (per sample)  : {overall_std:.1%}")

    # Per-sample breakdown — shows which samples vary most across rotations.
    print(f"\n  {'Sample':>8}  {'Mean acc':>10}  {'# rotations':>12}")
    for sid in sorted(per_sample):
        vals = per_sample[sid]
        m    = statistics.mean(vals)
        print(f"  sample{sid:>2}  {m:>10.1%}  {len(vals):>12}")


if __name__ == "__main__":
    main()
