"""Render the simple pipeline diagram — the same flow shown in README.md.

The README and docs/pipeline.md carry this as mermaid, which GitHub renders in
.md files but NOT inside .ipynb files. The notebooks embed this PNG instead, so
the diagram survives the trip to the repo.

Both the shape and the colours follow the README's mermaid source: one chain
down to stage 3, then a fork — Path A scores stage 3 directly, while the rules
branch continues to stage 4 and Path B.

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
BLUE  = "#3C6FA8"

p = argparse.ArgumentParser(description=__doc__)
# 300 dpi: the image is displayed around 700px wide but viewed on HiDPI screens
# and zoomed into, where 150 dpi visibly softened the text.
p.add_argument("--dpi", type=int, default=300)
p.add_argument("--out", type=Path, default=Path("Report-doc/pipeline_simple.png"))
args = p.parse_args()

fig, ax = plt.subplots(figsize=(7.0, 9.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 142)
ax.axis("off")

H, GAP = 9.2, 3.6        # box height, vertical gap between rows
CW, CX = 46, 50          # chain box width, centre
BW = 40                  # branch box width
LX, RX = 25, 75          # branch centres, left and right of the fork


def box(cx, y, w, *, face="#FFFFFF", edge=SLATE, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y), w, H, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))


def label(cx, y, main, sub=None, *, color="#141414", size=9.6):
    """Bold line, with an optional smaller line beneath it."""
    if sub is None:
        ax.text(cx, y + H / 2, main, fontsize=size, color=color, fontweight="bold",
                ha="center", va="center", zorder=5)
    else:
        ax.text(cx, y + H / 2 + 1.9, main, fontsize=size, color=color,
                fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(cx, y + H / 2 - 2.1, sub, fontsize=7.2, color=GREY,
                ha="center", va="center", zorder=5)


def down(y_from, y_to, cx=CX, color=NAVY):
    """Straight arrow between two stacked boxes."""
    ax.add_patch(FancyArrowPatch(
        (cx, y_from), (cx, y_to), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.5, color=color, shrinkA=0, shrinkB=0, zorder=3))


def fork(y_from, x_from, y_to, x_to, color=NAVY):
    """Elbow arrow: down out of the parent, across, then down into the child."""
    mid = (y_from + y_to) / 2
    ax.plot([x_from, x_from], [y_from, mid], color=color, linewidth=1.5, zorder=1)
    ax.plot([x_from, x_to], [mid, mid], color=color, linewidth=1.5, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (x_to, mid), (x_to, y_to), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.5, color=color, shrinkA=0, shrinkB=0, zorder=3))


# ── The chain, down to stage 3 ───────────────────────────────────────────
chain = [
    ("DMP PDF",             None,                                      NAVY,      NAVY,  "#FFFFFF"),
    ("Read the PDF",        "pdfplumber · Docling · LightOnOCR",       LIGHT,     SLATE, INK),
    ("1. Text blocks",      None,                                      "#FFFFFF", TEAL,  "#141414"),
    ("Label each block",    "llama3.1:8b · gemma4:e4b · llama3.3:70b", LIGHT,     SLATE, INK),
    ("2. Labeled blocks",   None,                                      "#FFFFFF", NAVY,  "#141414"),
    ("Build the structure", None,                                      LIGHT,     SLATE, INK),
    ("3. Structured JSON",  None,                                      "#FFFFFF", NAVY,  "#141414"),
]

y = 142 - 2 - H
for i, (main, sub, face, edge, ink) in enumerate(chain):
    box(CX, y, CW, face=face, edge=edge, lw=1.8 if edge in (TEAL, NAVY) and face == "#FFFFFF" else 1.4)
    label(CX, y, main, sub, color=ink)
    if i:
        down(prev, y + H)
    prev = y
    y -= H + GAP

s3_bottom = prev

# ── The fork ─────────────────────────────────────────────────────────────
# Left: Path A scores stage 3 as produced. Right: the rules branch continues
# to stage 4, which Path B scores. Same split the README's mermaid draws.
y_branch = s3_bottom - (H + GAP)

fork(s3_bottom, CX, y_branch + H, LX, color=BLUE)
box(LX, y_branch, BW, face="#EDF3FA", edge=BLUE, lw=1.8)
label(LX, y_branch, "Path A", "score vs old annotation", color=BLUE, size=9.4)

fork(s3_bottom, CX, y_branch + H, RX, color=AMBER)
box(RX, y_branch, BW, face=LIGHT, edge=SLATE)
label(RX, y_branch, "Apply the rules", "Rules.xlsx", color=INK)

y_s4 = y_branch - (H + GAP)
down(y_branch, y_s4 + H, cx=RX, color=AMBER)
box(RX, y_s4, BW, face="#FFFFFF", edge=AMBER, lw=1.8)
label(RX, y_s4, "4. Final JSON")

y_pb = y_s4 - (H + GAP)
down(y_s4, y_pb + H, cx=RX, color=AMBER)
box(RX, y_pb, BW, face="#FDF4E9", edge=AMBER, lw=1.8)
label(RX, y_pb, "Path B", "score vs new annotation", color=AMBER, size=9.4)

ax.text(CX, y_pb - 6.5, "Each numbered box is written to disk",
        fontsize=7.6, color=GREY, style="italic", ha="center", va="center")

fig.tight_layout(pad=0.2)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
print(f"saved -> {args.out}  ({args.dpi} dpi, {args.out.stat().st_size / 1024:.0f} KB)")
