"""One figure for two samples: the prompt-version summary heatmap (three models together,
count and share of the annotated fields) drawn as one panel per sample, side by side.

Same scoring and colours as the version-comparison notebook's summary cell; this script
exists because that notebook is one sample at a time.

    python scripts/build/make_rda_samples_summary.py              # samples 3 and 14, v9-v11
    python scripts/build/make_rda_samples_summary.py --samples 3 14 --versions v9 v10 v11
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from dmpbridge.rda import evaluate as E

RUNS_DIR = Path("notebooks/rda-experiments/runs")
OUT_DIR = Path("notebooks/rda-experiments")
MODELS = ["llama3.1-8b", "gemma4-e4b", "llama3.3-70b"]
OUTCOMES = ["Correct", "Mismatch", "Missed"]
COLOUR = {"Correct": "#238b45", "Mismatch": "#e34948", "Missed": "#898781"}
INK, SURFACE = "#0b0b0b", "#fcfcfb"
COLS = [("Correct ↑", "Correct"), ("Mismatch ↓", "Mismatch"), ("Missed ↓", "Missed"), ("Prompt size", None)]


def prompt_size(version: str, sample: int) -> str:
    """Tokens sent to llama3.1:8b for this sample, as the version-comparison notebook shows it."""
    folder = RUNS_DIR / f"{version}-sample{sample}"
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    sent = next((t["tokens_sent"] for k, t in info["timings"].items()
                 if k.startswith("llama3.1:8b") and f"sample{sample}" in k), None)
    prompt = (folder / "prompt.txt").read_text(encoding="utf-8")
    with_schema = "{schema_text}" in prompt or "maDMP JSON schema" in prompt
    size = f"~{round(sent, -2):,} tokens" if sent else "?"
    return size + (chr(10) + "(schema text in prompt)" if with_schema else "")


def totals_for(sample: int, versions: list[str]) -> tuple[dict[str, Counter], int]:
    """Correct/mismatch/missed summed over the three models, per version, and the field count."""
    ref = E.reference_path(sample)
    totals, n_fields = {}, 0
    for v in versions:
        c = Counter()
        for m in MODELS:
            f = RUNS_DIR / f"{v}-sample{sample}" / f"sample{sample}.rda.{m}.json"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            _, _, outcomes = E.evaluate_file(f, ref)
            c.update(outcomes.values())
            n_fields = len(outcomes)
        totals[v] = c
    return totals, n_fields


def draw_panel(ax, totals, n_fields, versions, sample):
    n_total = len(MODELS) * n_fields
    surface = np.array(mcolors.to_rgb(SURFACE))
    img = np.ones((len(versions), len(COLS), 3)) * surface
    for i, v in enumerate(versions):
        for j, (_, outcome) in enumerate(COLS):
            if not outcome:
                continue
            n = totals[v][outcome]
            strength = 0.12 + 0.88 * (n / n_total) if n else 0.04
            img[i, j] = surface * (1 - strength) + np.array(mcolors.to_rgb(COLOUR[outcome])) * strength
    ax.imshow(img, aspect="auto")
    for i, v in enumerate(versions):
        for j, (_, outcome) in enumerate(COLS):
            if not outcome:
                ax.text(j - 0.45, i, prompt_size(v, sample), ha="left", va="center", fontsize=11, color=INK)
                continue
            n = totals[v][outcome]
            share = n / n_total
            ink = "white" if share > 0.5 else INK
            if n:
                ax.text(j - 0.07, i, str(n), ha="right", va="center", fontsize=15, fontweight="bold", color=ink)
                ax.text(j + 0.04, i, f"{share:.0%}", ha="left", va="center", fontsize=11.5, color=ink)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=14, color="#c9c8c3")
    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(versions, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([label for label, _ in COLS], fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for i in range(len(versions) + 1):
        ax.axhline(i - 0.5, color=SURFACE, linewidth=4, zorder=3)
    for j in range(len(COLS) + 1):
        ax.axvline(j - 0.5, color=SURFACE, linewidth=4, zorder=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Sample {sample}  ·  {n_fields} annotated fields × {len(MODELS)} models = {n_total}",
                 loc="center", fontsize=12.5, fontweight="bold", pad=34)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, nargs="+", default=[3, 14])
    ap.add_argument("--versions", nargs="+", default=["v9", "v10", "v11"])
    args = ap.parse_args()

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK})
    fig, axes = plt.subplots(1, len(args.samples), figsize=(8.4 * len(args.samples), 0.8 * len(args.versions) + 2.1))
    for ax, sample in zip(np.atleast_1d(axes), args.samples):
        totals, n_fields = totals_for(sample, args.versions)
        draw_panel(ax, totals, n_fields, args.versions, sample)
        print(f"sample {sample}: " + "; ".join(
            f"{v} {totals[v]['Correct']}/{totals[v]['Mismatch']}/{totals[v]['Missed']}" for v in args.versions))
    fig.suptitle("Prompt versions, three models together: count and share of the annotated fields",
                 x=0.5, ha="center", fontsize=15, fontweight="bold", y=1.07)
    fig.subplots_adjust(wspace=0.28, top=0.80)
    out = OUT_DIR / ("version_comparison_summary_samples" + "_".join(map(str, args.samples)) + ".png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
