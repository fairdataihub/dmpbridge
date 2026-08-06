"""Render the simple pipeline diagram — the same flow shown in README.md.

The README and docs/pipeline.md carry this as mermaid, which GitHub renders in
.md files but NOT inside .ipynb files. The notebooks embed this PNG instead, so
the diagram survives the trip to the repo.

Colours match the README's mermaid classDefs, so the two read as one diagram.

    python make_pipeline_simple.py [--dpi 150] [--out PATH]

For the detailed version used in the Word report, see make_pipeline_diagram.py.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY  = "#1E406E"
TEAL  = "#0F766E"
AMBER = "#B45309"
SLATE = "#94A3B8"
INK   = "#334155"
GREY  = "#5B6470"
LIGHT = "#F4F7FB"

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dpi", type=int, default=150)
p.add_argument("--out", type=Path, default=Path("Report-doc/pipeline_simple.png"))
args = p.parse_args()

# One column for the flow, one for the two scores branching off it.
CX, CW = 16, 44          # chain column: x, width
RX, RW = 66, 32          # score column

fig, ax = plt.subplots(figsize=(6.6, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 116)
ax.axis("off")


def box(x, y, w, h, *, face="#FFFFFF", edge=SLATE, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))


def label(x, y, main, sub=None, *, color="#141414", size=9.6, subcolor=GREY):
    """Bold line, with an optional smaller line beneath it."""
    if sub is None:
        ax.text(x, y, main, fontsize=size, color=color, fontweight="bold",
                ha="center", va="center", zorder=5)
    else:
        ax.text(x, y + 1.9, main, fontsize=size, color=color, fontweight="bold",
                ha="center", va="center", zorder=5)
        ax.text(x, y - 2.1, sub, fontsize=7.2, color=subcolor,
                ha="center", va="center", zorder=5)


def arrow(y_from, y_to, x=CX + CW / 2, color=NAVY):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.5, color=color, shrinkA=0, shrinkB=0, zorder=3))


def connect(y, color):
    """Horizontal line from the chain column into the score column."""
    ax.plot([CX + CW, RX], [y, y], color=color, linewidth=1.5, zorder=1)


H = 9.2                  # box height
GAP = 3.6                # arrow gap between boxes
CXM = CX + CW / 2

# ── The chain, top to bottom ─────────────────────────────────────────────
rows = [
    ("PDF",   "DMP PDF",             None,                                        NAVY,  NAVY,  "#FFFFFF"),
    ("READ",  "Read the PDF",        "pdfplumber · Docling · LightOnOCR",         LIGHT, SLATE, INK),
    ("S1",    "1. Text blocks",      None,                                        "#FFFFFF", TEAL,  "#141414"),
    ("LABEL", "Label each block",    "llama3.1:8b · gemma4:e4b · llama3.3:70b",   LIGHT, SLATE, INK),
    ("S2",    "2. Labeled blocks",   None,                                        "#FFFFFF", NAVY,  "#141414"),
    ("BUILD", "Build the structure", None,                                        LIGHT, SLATE, INK),
    ("S3",    "3. Structured JSON",  None,                                        "#FFFFFF", NAVY,  "#141414"),
    ("RULES", "Apply the rules",     "Rules.xlsx",                                LIGHT, SLATE, INK),
    ("S4",    "4. Final JSON",       None,                                        "#FFFFFF", AMBER, "#141414"),
]

y = 116 - H - 2
ys = {}
for i, (key, main, sub, face, edge, ink) in enumerate(rows):
    box(CX, y, CW, H, face=face, edge=edge, lw=1.8 if sub is None and key != "PDF" else 1.4)
    label(CXM, y + H / 2, main, sub, color=ink)
    ys[key] = (y, y + H)
    if i:
        arrow(prev_bottom, y + H)
    prev_bottom = y
    y -= H + GAP

# ── The two scores ───────────────────────────────────────────────────────
# Path A leaves stage 3; Path B leaves stage 4. Drawn to the side rather than
# below, so the vertical chain still reads as one uninterrupted flow.
for key, name, face, edge, note in (
    ("S3", "Path A", "#EDF3FA", "#3C6FA8", "score vs old annotation"),
    ("S4", "Path B", "#FDF4E9", AMBER,     "score vs new annotation"),
):
    bottom, top = ys[key]
    mid = (bottom + top) / 2
    connect(mid, edge)
    box(RX, bottom, RW, H, face=face, edge=edge, lw=1.8)
    label(RX + RW / 2, mid, name, note, color=edge, size=9.4, subcolor=GREY)

ax.text(CXM, ys["S4"][0] - 6.4,
        "Each numbered box is written to disk",
        fontsize=7.6, color=GREY, style="italic", ha="center", va="center")

fig.tight_layout(pad=0.2)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
print(f"saved -> {args.out}  ({args.dpi} dpi, {args.out.stat().st_size / 1024:.0f} KB)")
