"""Tables and charts for one version and across versions (needs pandas + matplotlib)."""
import difflib
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# The repo's palette: a validated blue / red pair plus a neutral gray for "missed",
# and the same model slot order every other notebook uses.
VERDICT_COLOUR = {"Correct": "#2a78d6", "Hallucinated": "#e34948", "Missed": "#898781"}
MODEL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#d8d8d4", "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.grid": False, "legend.frameon": False,
})
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)


def model_colour(models):
    return dict(zip(models, MODEL_COLOURS))


def _tidy(ax, axis):
    ax.grid(axis=axis, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ── One version ───────────────────────────────────────────────────────────────

def worked_example(summary):
    """The arithmetic behind precision, recall and F1, with the first model's numbers."""
    m = summary.index[0]
    c, out, n_ref, miss = (int(summary.loc[m, k]) for k in ("correct", "fields output", "fields in reference", "missed"))
    print(f"Worked example, {m}:")
    print(f"  precision = correct / fields output          = {c} / {out} = {summary.loc[m, 'precision']}")
    print(f"  recall    = (reference - missed) / reference = ({n_ref} - {miss}) / {n_ref} = {summary.loc[m, 'recall']}")
    print(f"  F1        = 2 * precision * recall / (precision + recall) = {summary.loc[m, 'f1']}")


def plot_version(summary, models, title, save_to=None):
    """Left: each model's fields as correct / hallucinated / missed shares.
    Right: precision, recall and F1 per model."""
    scored = [m for m in models if m in summary.index]
    colour = model_colour(models)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

    labels = [f"{m}\n{int(summary.loc[m, 'correct'])} correct · {int(summary.loc[m, 'hallucinated'])} hallucinated · "
              f"{int(summary.loc[m, 'missed'])} missed" for m in scored]
    totals = {m: int(summary.loc[m, ["correct", "hallucinated", "missed"]].sum()) for m in scored}
    left = [0.0] * len(scored)
    for v in ("Correct", "Hallucinated", "Missed"):
        vals = [summary.loc[m, v.lower()] / totals[m] for m in scored]
        bars = ax1.barh(labels, vals, left=left, height=0.52, color=VERDICT_COLOUR[v],
                        edgecolor=SURFACE, linewidth=2, label=v)
        for bar, val in zip(bars, vals):
            if val >= 0.05:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                         ha="center", va="center", fontsize=8.5, color="white" if v != "Missed" else INK)
        left = [l + x for l, x in zip(left, vals)]
    ax1.set_xlim(0, 1)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_title(f"{title}: fields correct, hallucinated, missed")
    ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _tidy(ax1, "x")
    ax1.spines["left"].set_visible(False)

    shown = ["precision", "recall", "f1"]
    w = 0.8 / len(scored)
    for k, m in enumerate(scored):
        xs = [i + (k - (len(scored) - 1) / 2) * w for i in range(len(shown))]
        bars = ax2.bar(xs, [summary.loc[m, s] for s in shown], width=w * 0.92, color=colour[m],
                       edgecolor=SURFACE, linewidth=2, label=m)
        ax2.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=MUTED)
    ax2.set_xticks(range(len(shown)))
    ax2.set_xticklabels(["Precision", "Recall", "F1"])
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_title("Precision, recall and F1")
    ax2.legend(ncol=len(scored), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _tidy(ax2, "y")

    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.show()


def show_fields(details, models):
    """Every field a model wrote, next to the reference, hallucinated rows first."""
    from IPython.display import display

    order = {"Hallucinated": 0, "Correct": 1}
    for m in models:
        d = details[(details["model"] == m) & (details["verdict"] != "Missed")].copy()
        if d.empty:
            continue
        d = d.sort_values(["verdict", "model field"], key=lambda col: col.map(order) if col.name == "verdict" else col)
        counts = details[details["model"] == m]["verdict"].value_counts()
        print(f"\n{'=' * 100}\n{m}: {counts.get('Correct', 0)} correct, {counts.get('Hallucinated', 0)} hallucinated, "
              f"{counts.get('Missed', 0)} missed\n{'=' * 100}")
        display(d[["model field", "model value", "reference value", "verdict", "reason"]]
                .rename(columns={"model field": "field"}).reset_index(drop=True))


# ── Across versions ───────────────────────────────────────────────────────────

def load_history(runs_dir):
    """One row per (version, model) for every scored version, oldest first."""
    rows = []
    for d in Path(runs_dir).iterdir():
        if d.is_dir() and (d / "scores.json").exists():
            info = json.loads((d / "scores.json").read_text(encoding="utf-8"))
            for m, s in info["scores"].items():
                rows.append({"version": d.name, "date": info["date"], "model": m, **s})
    history = pd.DataFrame(rows).sort_values(["date", "version"]).reset_index(drop=True)
    history.attrs["versions"] = list(dict.fromkeys(history["version"]))
    history.attrs["runs_dir"] = str(runs_dir)
    return history


def show_history(history, models):
    """The scores per version and model, and why a model has no score in a version."""
    from IPython.display import display

    versions = history.attrs["versions"]
    print("versions, oldest first:", ", ".join(versions))
    display(history.set_index(["version", "model"])[["correct", "hallucinated", "missed", "precision", "recall", "f1"]].loc[versions])
    print("\nF1 per version and model:")
    display(history.pivot(index="version", columns="model", values="f1").loc[versions]
            .reindex(columns=[m for m in models if m in set(history["model"])]))
    runs_dir = Path(history.attrs.get("runs_dir", "data/output/rda/runs"))
    for v in versions:
        p = runs_dir / v / "run.json"
        info = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        for m, why in info.get("not_run", {}).items():
            print(f"{v}: no score for {m} - {why}")


def plot_history(history, models, save_to=None):
    """F1 and recall across versions, one line per model."""
    versions = history.attrs["versions"]
    colour = model_colour(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, metric, name in zip(axes, ("f1", "recall"), ("F1", "Recall")):
        for k, m in enumerate(models):
            h = history[history["model"] == m].set_index("version").reindex(versions)
            if h[metric].notna().any():
                ax.plot(range(len(versions)), h[metric], marker="o", markersize=7, linewidth=2, color=colour[m], label=m)
                for x, y in enumerate(h[metric]):
                    if pd.notna(y):                           # labels above / below / right so models don't collide
                        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", ha="center", fontsize=8,
                                    color=colour[m], xytext=[(0, 7), (0, -13), (14, -3)][k % 3])
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{name} by version")
        _tidy(ax, "y")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.show()


def _status_per_field(df):
    """One mark per reference field: Correct beats Hallucinated beats Missed."""
    rank = {"Correct": 0, "Hallucinated": 1, "Missed": 2}
    return df.assign(r=df["verdict"].map(rank)).sort_values("r").drop_duplicates("field").set_index("field")["verdict"]


def _some(fields, n=6):
    return ", ".join(fields[:n]) + (f", ... and {len(fields) - n} more" if len(fields) > n else "")


def show_changes(version, history, summary, details):
    """What changed, field by field and in the prompt, since the version before this one."""
    versions = history.attrs["versions"]
    i = versions.index(version.name)
    if i == 0:
        print(f"{version.name!r} is the only scored version - nothing to compare with yet.")
        return
    previous = versions[i - 1]
    print(f"What changed from {previous!r} to {version.name!r}\n")
    before_all = pd.read_csv(version.runs_dir / previous / "details.csv")
    for m in summary.index:
        if m not in set(before_all["model"]):
            print(f"{m}: not in {previous!r}, so no comparison\n")
            continue
        before = _status_per_field(before_all[before_all["model"] == m])
        after = _status_per_field(details[details["model"] == m])
        both = pd.concat([before.rename("before"), after.rename("after")], axis=1).fillna("Missed")
        changed = both[both["before"] != both["after"]]
        fixed = changed[changed["after"] == "Correct"]
        broken = changed[changed["before"] == "Correct"]
        other = len(changed) - len(fixed) - len(broken)
        prev = history[(history["version"] == previous) & (history["model"] == m)].iloc[0]
        now = summary.loc[m]
        print(f"{m}: F1 {prev['f1']:.3f} -> {now['f1']:.3f}   correct {int(prev['correct'])} -> {int(now['correct'])}, "
              f"hallucinated {int(prev['hallucinated'])} -> {int(now['hallucinated'])}, "
              f"missed {int(prev['missed'])} -> {int(now['missed'])}")
        print(f"   now right, was not before:  {len(fixed):3}   {_some(list(fixed.index))}")
        print(f"   was right, now not:         {len(broken):3}   {_some(list(broken.index))}")
        print(f"   wrong both times, differently (missed before and a wrong value now, or the reverse): {other}")
        print("   The full list is in details.csv in each version's folder.\n")

    print("Prompt change:\n")
    diff = list(difflib.unified_diff(
        (version.runs_dir / previous / "prompt.txt").read_text(encoding="utf-8").splitlines(),
        version.prompt_record.splitlines(), fromfile=previous, tofile=version.name, lineterm="", n=1))
    print("\n".join(diff) if diff else "   none - the same prompt (settings such as the model or context may differ)")
