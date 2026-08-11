"""Render the pipeline diagram used in section 3 of the report.

Defaults suit the Word report. The notebooks ask for a lighter copy, since the
image is embedded in every .ipynb and 230 dpi is far more than a 720px display
width needs:

    python make_pipeline_diagram.py --dpi 150 --out Report-doc/pipeline_diagram_web.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY  = "#1E406E"
BLUE  = "#3C6FA8"
TEAL  = "#0F766E"
AMBER = "#B45309"
GREY  = "#5B6470"
MONO  = "#6B7480"
LIGHT = "#F4F7FB"
EDGE  = "#CBD5E1"

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--dpi", type=int, default=230, help="output resolution (default: 230)")
_p.add_argument("--out", type=Path, default=Path("Report-doc/pipeline_diagram.png"),
                help="output path (default: Report-doc/pipeline_diagram.png)")
args = _p.parse_args()

# Layout columns
SX, SW = 3, 62          # stage column: x, width
EX, EW = 70, 27         # evaluation column

fig, ax = plt.subplots(figsize=(7.4, 8.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 116)
ax.axis("off")


def box(x, y, w, h, *, face="#FFFFFF", edge=EDGE, lw=1.2, r=1.4, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def text(x, y, s, *, size=8.4, color="#141414", weight="normal",
         ha="center", va="center", style="normal", mono=False):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style=style, zorder=5,
            family="DejaVu Sans Mono" if mono else "DejaVu Sans")


def arrow(x1, y1, x2, y2, *, color=NAVY, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color=color, linewidth=lw,
                                 shrinkA=0, shrinkB=0, zorder=3))


def elbow(x1, y1, x2, y2, *, color=BLUE, lw=1.4):
    """Horizontal then vertical, with the arrowhead on the final leg."""
    ax.plot([x1, x2], [y1, y1], color=color, lw=lw, zorder=3,
            solid_capstyle="round")
    arrow(x2, y1, x2, y2, color=color, lw=lw)


def stage(y, h, n, title, path, note, accent):
    """One numbered pipeline stage: coloured spine, title, path, note."""
    box(SX, y, SW, h, edge=accent, lw=1.6)
    ax.add_patch(FancyBboxPatch((SX, y), 2.4, h,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=accent, edgecolor=accent, zorder=3))
    text(SX + 5.5, y + h - 3.6, f"STAGE {n}", size=7.2, color=accent,
         weight="bold", ha="left")
    text(SX + 18.5, y + h - 3.6, title, size=9.2, weight="bold", ha="left")
    text(SX + 5.5, y + h - 8.2, path, size=6.9, color=MONO, ha="left", mono=True)
    text(SX + 5.5, y + 3.0, note, size=7.2, color=accent, ha="left", style="italic")


CX = SX + SW / 2        # centre of the stage column

# ── Input ────────────────────────────────────────────────────────────────
box(SX + 8, 108, SW - 16, 7, face=NAVY, edge=NAVY)
text(CX, 111.5, "DMP PDF   ·   10 documents, 23 pages", size=9.4,
     color="white", weight="bold")
arrow(CX, 108, CX, 103)
text(CX + 1.5, 105.5, "choose one extraction backend", size=7.0,
     color=GREY, style="italic", ha="left")

# ── Extractors ───────────────────────────────────────────────────────────
for i, (label, sub) in enumerate((("pdfplumber", "line-level"),
                                  ("Docling", "ML layout"),
                                  ("LightOnOCR", "vision OCR"))):
    x = SX + i * 21.3
    box(x, 94, 19.4, 8.4, edge=BLUE, lw=1.3)
    text(x + 9.7, 99.6, label, size=8.5, weight="bold", color=BLUE)
    text(x + 9.7, 96.2, sub, size=6.9, color=GREY)
    ax.plot([x + 9.7, x + 9.7, CX], [94, 91.6, 91.6], color=BLUE, lw=1.1, zorder=3)
arrow(CX, 91.6, CX, 88.5, color=BLUE, lw=1.3)

# ── Stage 1 ──────────────────────────────────────────────────────────────
stage(74, 14.5, 1, "Extracted blocks",
      "data/output/1_extracted/<extractor>/sampleN.json",
      "cached per extractor — computed once, reused by every model", TEAL)

arrow(CX, 74, CX, 67.5)
text(CX + 1.5, 72.4, "LLM  ·  whole-doc", size=7.6, weight="bold",
     color=NAVY, ha="left")
text(CX + 1.5, 69.4, "one call per document, temperature 0.0", size=6.9,
     color=GREY, ha="left")

# ── Stage 2 ──────────────────────────────────────────────────────────────
stage(53, 14.5, 2, "Labeled blocks",
      "data/output/2_labeled/<tag>/sampleN.json",
      "every block gets one of 5 labels + a confidence score", NAVY)

arrow(CX, 53, CX, 46.5)
text(CX + 1.5, 49.8, "converter.to_structured()", size=7.0, color=GREY, ha="left")

# ── Stage 3 ──────────────────────────────────────────────────────────────
stage(32, 14.5, 3, "Structured JSON",
      "data/output/3_structured/<tag>/sampleN.json",
      "nested schema: sections → questions → answers", NAVY)

arrow(CX, 32, CX, 25.5)
text(CX + 1.5, 29.6, "Rules.xlsx", size=7.4, weight="bold", color=AMBER, ha="left")
text(CX + 1.5, 26.9, "fill empty question.text", size=6.9, color=GREY, ha="left")

# ── Stage 4 ──────────────────────────────────────────────────────────────
stage(11, 14.5, 4, "Final JSON",
      "data/output/4_final/<tag>/sampleN.json",
      "source order: section.title → description → title", AMBER)

# ── Evaluation branches ──────────────────────────────────────────────────
box(EX, 33.5, EW, 11.5, face="#EDF3FA", edge=BLUE, lw=1.5)
text(EX + EW / 2, 41.4, "PATH A", size=8.6, weight="bold", color=BLUE)
text(EX + EW / 2, 37.9, "scored against", size=6.9, color=GREY)
text(EX + EW / 2, 35.4, "ground_truth_old_version", size=6.8, color=BLUE, mono=True)
elbow(SX + SW, 39.2, EX, 39.2, color=BLUE)

box(EX, 12.5, EW, 11.5, face="#FDF4E9", edge=AMBER, lw=1.5)
text(EX + EW / 2, 20.4, "PATH B", size=8.6, weight="bold", color=AMBER)
text(EX + EW / 2, 16.9, "scored against", size=6.9, color=GREY)
text(EX + EW / 2, 14.4, "ground_truth_new_version", size=6.8, color=AMBER, mono=True)
elbow(SX + SW, 18.2, EX, 18.2, color=AMBER)

# ── Shared scoring engine ────────────────────────────────────────────────
# An annotation about both paths, not a pipeline stage — deliberately drawn
# without arrows so it does not read as another step in the flow.
box(SX, 1.0, 97 - SX, 6.6, face=LIGHT, edge=EDGE, lw=1.1)
text((SX + 97) / 2, 5.4,
     "Both paths use one scoring engine — nothing differs but the reference data",
     size=7.6, weight="bold", color=NAVY)
text((SX + 97) / 2, 2.8,
     "greedy match by token containment ≥ 0.75   ·   micro-averaged precision / recall / F1",
     size=7.0, color=GREY)

fig.tight_layout(pad=0.25)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
print(f"saved -> {args.out}  ({args.dpi} dpi, {args.out.stat().st_size / 1024:.0f} KB)")
