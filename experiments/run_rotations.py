"""Run all leave-2-out rotations and report mean ± std gold_acc.

Results are reported per (model, extractor) combination so the full
extraction × model matrix is visible in one table.

Usage
-----
    python experiments/run_rotations.py                     # run + evaluate all
    python experiments/run_rotations.py --evaluate-only     # skip inference
    python experiments/run_rotations.py --dir experiments/rotations
    python experiments/run_rotations.py --model llama3.1:8b --extractor pdfplumber
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpbridge.evaluation.experiment import Experiment
from dmpbridge.evaluation.evaluate import extract_gold, evaluate_sample, gold_metrics, MANUAL_DIR

ROTATIONS_DIR = Path(__file__).parent / "rotations"


def _gold_acc_for(cfg, model: str, extractor: str) -> dict[int, float]:
    """Return {sample_id: gold_acc} for one (model, extractor) pair."""
    few_shot = set(cfg.few_shot_samples)
    eval_ids = set(cfg.eval_samples) if cfg.eval_samples else (set(cfg.sample_range) - few_shot)

    out_dir = Path(cfg.out_dir) / cfg.tag_for(model, extractor)
    accs: dict[int, float] = {}

    for sid in sorted(eval_ids):
        pred_path = out_dir / f"sample{sid}.json"
        gold_path = MANUAL_DIR / f"sample{sid}_dmp.json"
        if not pred_path.exists() or not gold_path.exists():
            continue
        gold_pairs        = extract_gold(gold_path)
        confusion         = evaluate_sample(pred_path, gold_pairs)
        correct, _, total = gold_metrics(confusion)
        if total > 0:
            accs[sid] = correct / total
    return accs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all leave-2-out rotations and report results."
    )
    ap.add_argument(
        "--dir", default=str(ROTATIONS_DIR),
        help="Directory containing rotation YAML configs (default: experiments/rotations).",
    )
    ap.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip inference — only report accuracy for existing result files.",
    )
    ap.add_argument(
        "--model", default=None,
        help="Filter to a single model (e.g. llama3.1:8b). Runs all models if omitted.",
    )
    ap.add_argument(
        "--extractor", default=None,
        choices=["pdfplumber", "docling", "lighton"],
        help="Filter to a single extractor. Runs all extractors if omitted.",
    )
    args = ap.parse_args()

    rotation_dir = Path(args.dir)
    yaml_files   = sorted(rotation_dir.glob("*.yaml"))

    if not yaml_files:
        print(f"No YAML files found in {rotation_dir}")
        sys.exit(1)

    print("\nLeave-2-out rotation evaluation")
    print(f"Configs   : {rotation_dir}")
    print(f"Rotations : {len(yaml_files)}\n")

    # {(model, extractor): [gold_acc per sample across all rotations]}
    all_accs:   dict[tuple, list[float]] = {}
    per_sample: dict[tuple, dict[int, list[float]]] = {}

    print(
        f"  {'Rotation':<46}  {'Model':<16}  {'Extractor':<12}"
        f"  {'Few-shot':>10}  {'N eval':>6}  {'Gold acc':>9}"
    )
    print("  " + "-" * 108)

    for yaml_path in yaml_files:
        exp = Experiment.from_yaml(yaml_path)
        cfg = exp.config

        # Apply CLI filters
        models     = [m for m in cfg.models     if args.model     is None or m     == args.model]
        extractors = [e for e in cfg.extractors if args.extractor is None or e == args.extractor]

        if not args.evaluate_only:
            # Run only the filtered subset
            for model in models:
                for extractor in extractors:
                    exp._run_combination(model, extractor)

        few_str = str(cfg.few_shot_samples)

        for model in models:
            for extractor in extractors:
                accs = _gold_acc_for(cfg, model, extractor)
                if not accs:
                    print(
                        f"  {cfg.name:<46}  {model:<16}  {extractor:<12}"
                        f"  {few_str:>10}  {'—':>6}  {'no results':>9}"
                    )
                    continue

                rot_mean = sum(accs.values()) / len(accs)
                key = (model, extractor)
                all_accs.setdefault(key, []).extend(accs.values())
                for sid, acc in accs.items():
                    per_sample.setdefault(key, {}).setdefault(sid, []).append(acc)

                print(
                    f"  {cfg.name:<46}  {model:<16}  {extractor:<12}"
                    f"  {few_str:>10}  {len(accs):>6}  {rot_mean:>8.1%}"
                )

    if not all_accs:
        print("\nNo results to summarise — run without --evaluate-only first.")
        return

    print("  " + "-" * 108)
    print("\n  Summary per (model, extractor):\n")
    print(f"  {'Model':<16}  {'Extractor':<12}  {'Mean gold acc':>14}  {'Std dev':>8}")
    print("  " + "-" * 58)

    for (model, extractor), accs in sorted(all_accs.items()):
        mean = statistics.mean(accs)
        std  = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {model:<16}  {extractor:<12}  {mean:>13.1%}  {std:>7.1%}")

    print()
    # Per-sample variance — shows which samples are hardest across rotations
    print(f"  {'Sample':>8}  {'Model':<16}  {'Extractor':<12}  {'Mean':>8}  {'Rotations':>10}")
    for (model, extractor), samples in sorted(per_sample.items()):
        for sid in sorted(samples):
            vals = samples[sid]
            m    = statistics.mean(vals)
            print(f"  sample{sid:>2}  {model:<16}  {extractor:<12}  {m:>7.1%}  {len(vals):>10}")


if __name__ == "__main__":
    main()
