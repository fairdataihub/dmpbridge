"""Required vs recommended: for one run (prompt version + sample), how many of the annotated
fields each model got correct, split by the marker the prompt's form gives that field -
[REQUIRED] (incl. "REQUIRED if ... is named", since an annotated field means it was named)
or [RECOMMENDED]. Annotated fields the form does not have (e.g. contributors) are reported
separately. Prints a markdown table and saves it as a PNG in the run folder.

    python scripts/build/make_rda_required_table.py                      # v11, sample 3
    python scripts/build/make_rda_required_table.py --version v11 --sample 14
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmpbridge.rda import evaluate as E

RUNS_DIR = Path("notebooks/rda-experiments/runs")
PROMPTS_DIR = Path("scripts/build/prompts")
MODELS = ["llama3.1-8b", "gemma4-e4b", "llama3.3-70b"]
NAME = {"llama3.1-8b": "Llama 3.1 8B", "gemma4-e4b": "Gemma 4 e4b", "llama3.3-70b": "Llama 3.3 70B"}
INK, SURFACE = "#0b0b0b", "#fcfcfb"

_KEY_OPEN = re.compile(r'^"(?P<key>[^"]+)":\s*(?P<open>[\[{]+)\s*$')
_KEY_MARK = re.compile(r'^"(?P<key>[^"]+)":\s*"\[(?P<mark>REQUIRED|RECOMMENDED)')


def form_markers(prompt_text: str) -> dict[str, str]:
    """Dotted field path (list levels collapsed) -> 'required' | 'recommended', read from the form."""
    start = prompt_text.index("JSON FORM")
    stack: list[tuple[str | None, int]] = []
    markers: dict[str, str] = {}
    for raw in prompt_text[start:].splitlines()[1:]:
        line = raw.strip()
        if not line:
            continue
        if m := _KEY_MARK.match(line):
            path = ".".join(k for k, _ in stack if k) + "." + m["key"]
            markers[path] = m["mark"].lower()
        elif m := _KEY_OPEN.match(line):
            stack.append((m["key"], len(m["open"])))
        elif re.fullmatch(r"[\[{]+", line):
            stack.append((None, len(line)))
        elif re.fullmatch(r"[\]}]+,?", line):
            closers = len(line.rstrip(","))
            while closers > 0 and stack:
                _, n = stack.pop()
                closers -= n
            if closers == 0 and not stack:
                break
    return markers


def classify(field: str, markers: dict[str, str]) -> str:
    return markers.get(re.sub(r"\[\d+\]", "", field), "not in form")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v11")
    ap.add_argument("--sample", type=int, default=3)
    args = ap.parse_args()
    run = f"{args.version}-sample{args.sample}"

    markers = form_markers((PROMPTS_DIR / f"rda_narrative_{args.version}.txt").read_text(encoding="utf-8"))
    ref = E.reference_path(args.sample)
    rows, totals = [], None
    for m in MODELS:
        f = RUNS_DIR / run / f"sample{args.sample}.rda.{m}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        _, _, outcomes = E.evaluate_file(f, ref)
        groups = {"required": [0, 0], "recommended": [0, 0], "not in form": [0, 0]}
        for field, outcome in outcomes.items():
            g = groups[classify(field, markers)]
            g[1] += 1
            g[0] += outcome == "Correct"
        rows.append((NAME[m], groups))
        totals = {k: v[1] for k, v in groups.items()}

    extra = totals["not in form"]
    heads = ["Model", f"Required (total {totals['required']})", f"Recommended (total {totals['recommended']})"]
    keys = ["required", "recommended"]
    if extra:
        heads.append(f"Not in the form (total {extra})")
        keys.append("not in form")
    cells = [[name] + [f"{g[k][0]} / {g[k][1]}  ({g[k][0] / g[k][1]:.0%})" if g[k][1] else "–" for k in keys]
             for name, g in rows]

    print(f"{run}: {sum(totals.values())} annotated fields; markers from the {args.version} form\n")
    print("| " + " | ".join(heads) + " |")
    print("|" + "|".join("---" for _ in heads) + "|")
    for c in cells:
        print("| " + " | ".join(c) + " |")

    fig, ax = plt.subplots(figsize=(3.2 * len(heads), 0.55 * len(cells) + 1.5))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=heads, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#b9b8b3")
        cell.set_facecolor(SURFACE)
        cell.PAD = 0.04
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title(f"Fields correct by marker - sample {args.sample}, prompt {args.version}",
                 loc="left", fontsize=13, fontweight="bold", pad=10)
    out = RUNS_DIR / run / "required_vs_recommended.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
