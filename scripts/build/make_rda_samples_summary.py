"""One figure for two samples: prompt version (rows) against model (columns), showing how many
of the annotated fields each got CORRECT - count and share - one panel per sample, side by side,
with the prompt size as a last column.

Same scoring and colours as the version-comparison notebook; this script exists because that
notebook is one sample at a time.

    python scripts/build/make_rda_samples_summary.py              # samples 3 and 14, v9-v11
    python scripts/build/make_rda_samples_summary.py --samples 3 14 --versions v9 v10 v11
    python scripts/build/make_rda_samples_summary.py --no-prompt-size
"""
from __future__ import annotations

import argparse
import json
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
NAME = {"llama3.1-8b": "Llama 3.1 8B", "gemma4-e4b": "Gemma 4 e4b", "llama3.3-70b": "Llama 3.3 70B"}
GREEN = "#238b45"
INK, SURFACE = "#0b0b0b", "#fcfcfb"


def correct_for(sample: int, versions: list[str]) -> tuple[dict[tuple[str, str], int], int]:
    """Correct count per (version, model) and the number of annotated fields."""
    ref = E.reference_path(sample)
    correct, n_fields = {}, 0
    for v in versions:
        for m in MODELS:
            f = RUNS_DIR / f"{v}-sample{sample}" / f"sample{sample}.rda.{m}.json"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            _, _, outcomes = E.evaluate_file(f, ref)
            correct[(v, m)] = sum(o == "Correct" for o in outcomes.values())
            n_fields = len(outcomes)
    return correct, n_fields


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


def draw_panel(ax, correct, n_fields, versions, sample, with_size):
    cols = [NAME[m] for m in MODELS] + (["Prompt size"] if with_size else [])
    surface = np.array(mcolors.to_rgb(SURFACE))
    green = np.array(mcolors.to_rgb(GREEN))
    img = np.ones((len(versions), len(cols), 3)) * surface
    for i, v in enumerate(versions):
        for j, m in enumerate(MODELS):
            n = correct[(v, m)]
            strength = 0.12 + 0.88 * (n / n_fields) if n else 0.04
            img[i, j] = surface * (1 - strength) + green * strength
    ax.imshow(img, aspect="auto")
    for i, v in enumerate(versions):
        for j, m in enumerate(MODELS):
            n = correct[(v, m)]
            share = n / n_fields
            ink = "white" if share > 0.5 else INK
            if n:
                ax.text(j - 0.07, i, str(n), ha="right", va="center", fontsize=15, fontweight="bold", color=ink)
                ax.text(j + 0.04, i, f"{share:.0%}", ha="left", va="center", fontsize=11.5, color=ink)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=14, color="#c9c8c3")
        if with_size:
            ax.text(len(MODELS) - 0.45, i, prompt_size(v, sample), ha="left", va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(versions, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for i in range(len(versions) + 1):
        ax.axhline(i - 0.5, color=SURFACE, linewidth=4, zorder=3)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color=SURFACE, linewidth=4, zorder=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Sample {sample}  ·  {n_fields} annotated fields", loc="center",
                 fontsize=12.5, fontweight="bold", pad=34)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, nargs="+", default=[3, 14])
    ap.add_argument("--versions", nargs="+", default=["v9", "v10", "v11"])
    ap.add_argument("--no-prompt-size", dest="with_size", action="store_false",
                    help="leave out the prompt-size column")
    args = ap.parse_args()

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK})
    width = (8.4 if args.with_size else 6.2) * len(args.samples)
    fig, axes = plt.subplots(1, len(args.samples), figsize=(width, 0.8 * len(args.versions) + 2.1))
    for ax, sample in zip(np.atleast_1d(axes), args.samples):
        correct, n_fields = correct_for(sample, args.versions)
        draw_panel(ax, correct, n_fields, args.versions, sample, args.with_size)
        print(f"sample {sample} ({n_fields} fields): " + "; ".join(
            f"{v} " + "/".join(str(correct[(v, m)]) for m in MODELS) for v in args.versions))
    fig.suptitle("Fields extracted correctly, by prompt version and model",
                 x=0.5, ha="center", fontsize=15, fontweight="bold", y=1.07)
    fig.subplots_adjust(wspace=0.28, top=0.80)
    out = OUT_DIR / ("version_comparison_summary_samples" + "_".join(map(str, args.samples)) + ".png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
