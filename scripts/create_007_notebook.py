"""Create notebooks/007_qwen2-vl.ipynb."""
import json
from pathlib import Path


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

# ── Cell 0 — header ───────────────────────────────────────────────────────────
cells.append(md(
    "# Evaluation — Qwen2.5-VL 7B (Ollama — Vision Batch)\n\n"
    "## Experiment metadata\n\n"
    "| | Vision Batch (M09) |\n"
    "|---|---|\n"
    "| **Prompt strategy** | pdfplumber blocks + page PNG per page → Qwen2.5-VL |\n"
    "| **Input files** | `data/output/labeled/qwen2.5vl-7b_vision/sampleN.json` |\n"
    "| **Regenerate** | `dmpbridge-experiment experiments/qwen2-vl-vision.yaml` |\n"
    "| **Provider** | Ollama (local — free) |\n"
    "| **Model size** | 7B parameters, Q4_K_M |\n"
    "| **Page images** | `data/output/page_images/sampleN/page_NNN.png` (shared with M08) |\n"
    "| **Image sent to model** | Resized ≤ 900 px to fit 7B context window |\n\n"
    "**Labels (5):** `title` · `section.title` · `section.description` "
    "· `question.text` · `answer.text`  \n"
    "**Evaluation:** block-level accuracy + F1 against "
    "`data/input/ground_truth/sampleN_dmp.json`\n\n"
    "---\n\n"
    "## Sections\n"
    "1. Accuracy summary\n"
    "2. Per-sample accuracy\n"
    "3. Confusion matrix\n"
    "4. Precision / Recall / F1 per label\n"
    "5. Mislabeled blocks\n"
    "6. Comparison — Qwen2.5-VL 7B vs Claude Opus 4.8 (same vision-batch strategy)"
))

# ── Cell 1 — constants ────────────────────────────────────────────────────────
cells.append(code(
    'MODEL_NAME    = "qwen2.5vl-7b"\n'
    'MODEL_DISPLAY = "qwen2.5vl:7b"\n'
    'COLOR_VISION  = "#0e7490"   # teal\n'
    'COLOR_CLAUDE  = "#0369a1"   # blue (Claude vision for comparison)\n'
    'CMAP_VISION   = "GnBu"\n'
    'CMAP_CLAUDE   = "Blues"\n'
))

# ── Cell 2 — imports + load ───────────────────────────────────────────────────
cells.append(code(
    "from dmpbridge.evaluation.evaluate import (\n"
    "    extract_gold, evaluate_sample, match,\n"
    "    LABELS, SHORT, LLM_DIR, MANUAL_DIR, NO_MATCH,\n"
    "    load_method, compute_f1_rows, gold_metrics,\n"
    ")\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mticker\n"
    "import pandas as pd\n"
    "import seaborn as sns\n\n"
    "sns.set_theme(style=\"ticks\", palette=\"muted\")\n"
    "plt.rcParams.update({\n"
    '    "figure.facecolor":"white", "axes.facecolor":"white",\n'
    '    "axes.edgecolor":"#555555", "axes.linewidth":0.8,\n'
    '    "grid.color":"#dddddd",    "grid.linewidth":0.5,\n'
    '    "text.color":"#111111",    "axes.labelcolor":"#111111",\n'
    '    "xtick.color":"#111111",   "ytick.color":"#111111",\n'
    '    "font.size":11, "axes.titlesize":13, "axes.labelsize":11,\n'
    '    "xtick.labelsize":10, "ytick.labelsize":10,\n'
    '    "legend.fontsize":10, "legend.frameon":True,\n'
    '    "figure.dpi":120,\n'
    "})\n\n"
    "# Qwen vision batch\n"
    'df_v, conf_v, errs_v = load_method(f"{MODEL_NAME}_vision")\n\n'
    "# Claude vision batch for Section 6 comparison\n"
    'df_c, conf_c, errs_c = load_method("claude-opus-4-8_vision")\n'
    'CLAUDE_AVAILABLE = df_c is not None and not df_c.empty\n\n'
    "tc_v, tn_v = int(df_v[\"correct\"].sum()), int(df_v[\"total\"].sum())\n"
    'print(f"Qwen2.5-VL 7B vision batch : {tc_v}/{tn_v} ({tc_v/tn_v*100:.1f}%)")\n'
    "if CLAUDE_AVAILABLE:\n"
    "    tc_c, tn_c = int(df_c[\"correct\"].sum()), int(df_c[\"total\"].sum())\n"
    '    print(f"Claude Opus 4.8 vision batch: {tc_c}/{tn_c} ({tc_c/tn_c*100:.1f}%)")\n'
))

# ── Cell 3 — Section 1 markdown ───────────────────────────────────────────────
cells.append(md(
    "## 1 — Accuracy summary\n\n"
    "- **Accuracy** = blocks where the predicted label exactly matches gold.\n"
    "- Qwen2.5-VL 7B uses the same vision-batch strategy and page images as Claude M08 "
    "but runs locally at no API cost.\n"
    "- A single bad sample can pull the overall number down — check the per-sample view next."
))

# ── Cell 4 — accuracy table ───────────────────────────────────────────────────
cells.append(code(
    "def print_table(df, label):\n"
    "    hdr = f\"{'Sample':<12}  {'Total':>5}  {'Correct':>7}  {'Errors':>6}  {'Accuracy':>8}  Formula\"\n"
    "    print(f\"  {label}\")\n"
    "    print(\"  \" + \"-\" * len(hdr))\n"
    "    for row in df.itertuples():\n"
    "        print(f\"  {row.sample:<12}  {row.total:>5}  {row.correct:>7}  \"\n"
    "              f\"{row.errors:>6}  {row.accuracy*100:>7.1f}%  {row.formula}\")\n"
    "    print(\"  \" + \"-\" * len(hdr))\n"
    "    tn, tc = df[\"total\"].sum(), df[\"correct\"].sum()\n"
    "    print(f\"  {'TOTAL':<12}  {tn:>5}  {tc:>7}  {tn-tc:>6}  {tc/tn*100:>7.1f}%  {tc}/{tn}\\n\")\n"
    "\n"
    "print_table(df_v, f\"Vision Batch — {MODEL_DISPLAY}\")\n"
))

# ── Cell 5 — Section 2 markdown ───────────────────────────────────────────────
cells.append(md(
    "## 2 — Per-sample accuracy\n\n"
    "Colour coding: green ≥ 90%, amber 80–90%, red < 80%.  \n"
    "Samples below 80% are candidates for prompt tuning or fine-tuning."
))

# ── Cell 6 — per-sample bar chart ────────────────────────────────────────────
cells.append(code(
    "samples_ordered = df_v[\"sample\"].tolist()\n"
    "x = range(len(samples_ordered))\n"
    "acc_v = df_v.set_index(\"sample\").loc[samples_ordered, \"accuracy\"] * 100\n"
    "colors = [COLOR_VISION if v >= 90 else \"#f59e0b\" if v >= 80 else \"#dc2626\"\n"
    "          for v in acc_v]\n\n"
    "fig, ax = plt.subplots(figsize=(12, 5))\n"
    "bars = ax.bar(samples_ordered, acc_v, color=colors, width=0.6, edgecolor=\"white\")\n"
    "for bar, v in zip(bars, acc_v):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n"
    "            f\"{v:.1f}%\", ha=\"center\", va=\"bottom\", fontsize=9)\n"
    "avg_v = acc_v.mean()\n"
    "ax.axhline(avg_v, color=\"#444\", linestyle=\"--\", linewidth=1.0)\n"
    "ax.text(len(x)-0.45, avg_v + 0.7, f\"Mean = {avg_v:.1f}%\",\n"
    "        ha=\"right\", va=\"bottom\", fontsize=9.5, color=\"#444\")\n"
    "ax.set_xticks(list(x)); ax.set_xticklabels(samples_ordered, rotation=30)\n"
    "ax.set_ylim(40, 118); ax.set_ylabel(\"Block-level accuracy (%)\")\n"
    "ax.set_title(f\"Per-sample accuracy — {MODEL_DISPLAY} — Vision Batch\", pad=10)\n"
    "ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f\"{v:.0f}%\"))\n"
    "sns.despine(); plt.tight_layout(); plt.show()\n"
))

# ── Cell 7 — Section 3 markdown ───────────────────────────────────────────────
cells.append(md(
    "## 3 — Confusion matrix\n\n"
    "Read **row by row**: each row = a true label, each column = what the model predicted.  \n"
    "A perfect model has all mass on the diagonal.  \n"
    "The key pair to watch: `question.text` vs `section.description` — "
    "both are funder-written text; the boundary is structural, not lexical."
))

# ── Cell 8 — confusion matrix ─────────────────────────────────────────────────
cells.append(code(
    "def conf_to_df(conf):\n"
    "    m = pd.DataFrame(\n"
    "        [[conf.get(t, {}).get(p, 0) for p in LABELS] for t in LABELS],\n"
    "        index=SHORT, columns=SHORT)\n"
    "    return m, m.div(m.sum(axis=1).replace(0, 1), axis=0)\n\n"
    "mat_v, mat_vn = conf_to_df(conf_v)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "for ax, mat, fmt, title in [\n"
    "    (axes[0], mat_v,  \"d\",   \"Raw counts\"),\n"
    "    (axes[1], mat_vn, \".0%\", \"Recall per label (row-normalised)\"),\n"
    "]:\n"
    "    sns.heatmap(mat, annot=True, fmt=fmt, cmap=CMAP_VISION, linewidths=0.6,\n"
    "                linecolor=\"white\", ax=ax, cbar=False, annot_kws={\"size\": 11},\n"
    "                vmin=(0 if fmt == \".0%\" else None), vmax=(1 if fmt == \".0%\" else None))\n"
    "    ax.set_title(title, pad=8, fontsize=11)\n"
    "    ax.set_xlabel(\"Predicted\", labelpad=6)\n"
    "    ax.set_ylabel(\"True\", labelpad=6)\n"
    "plt.suptitle(f\"Confusion matrix — {MODEL_DISPLAY} — Vision Batch\", fontsize=13, y=1.02)\n"
    "plt.tight_layout(); plt.show()\n"
))

# ── Cell 9 — Section 4 markdown ───────────────────────────────────────────────
cells.append(md(
    "## 4 — Precision / Recall / F1 per label\n\n"
    "F1 balances precision and recall. Values below 80% indicate a label the model "
    "struggles to classify reliably.\n"
    "`question.text` and `section.description` are historically the hardest pair."
))

# ── Cell 10 — F1 chart ───────────────────────────────────────────────────────
cells.append(code(
    "df_f1_v = compute_f1_rows(conf_v)\n"
    "xi, bw = range(len(LABELS)), 0.26\n\n"
    "fig, ax = plt.subplots(figsize=(11, 5))\n"
    "ax.bar([i - bw for i in xi], df_f1_v[\"precision\"]*100, width=bw, label=\"Precision\",\n"
    "       color=COLOR_VISION, alpha=0.5, edgecolor=\"white\")\n"
    "ax.bar([i      for i in xi], df_f1_v[\"recall\"]*100,    width=bw, label=\"Recall\",\n"
    "       color=COLOR_VISION, alpha=0.75, edgecolor=\"white\")\n"
    "bars = ax.bar([i + bw for i in xi], df_f1_v[\"f1\"]*100, width=bw, label=\"F1\",\n"
    "              color=COLOR_VISION, edgecolor=\"white\")\n"
    "for bar, row in zip(bars, df_f1_v.itertuples()):\n"
    "    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,\n"
    "            f\"{row.f1*100:.0f}%\", ha=\"center\", va=\"bottom\",\n"
    "            fontsize=9, fontweight=\"bold\", color=COLOR_VISION)\n"
    "ax.set_xticks(list(xi)); ax.set_xticklabels(SHORT, fontsize=10)\n"
    "ax.set_ylim(0, 120); ax.set_ylabel(\"Score (%)\")\n"
    "ax.set_title(f\"Precision / Recall / F1 — {MODEL_DISPLAY} — Vision Batch\", pad=10)\n"
    "ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f\"{v:.0f}%\"))\n"
    "ax.legend(loc=\"upper right\", framealpha=0.9)\n"
    "sns.despine(); plt.tight_layout(); plt.show()\n\n"
    "display(df_f1_v.set_index(\"label\").style.format(\"{:.1%}\")\n"
    "        .background_gradient(subset=[\"f1\"], cmap=\"GnBu\"))\n"
))

# ── Cell 11 — Section 5 markdown ──────────────────────────────────────────────
cells.append(md(
    "## 5 — Mislabeled blocks\n\n"
    "Systematic errors (same true→pred pair, many blocks) signal a prompt or label issue.  \n"
    "Isolated errors are acceptable noise."
))

# ── Cell 12 — error breakdown ─────────────────────────────────────────────────
cells.append(code(
    "def error_breakdown(df_err, label):\n"
    "    print(f\"  {label}: {len(df_err)} mislabeled blocks\")\n"
    "    if df_err.empty:\n"
    "        print(\"  No errors.\\n\"); return\n"
    "    bd = (df_err.groupby([\"true\",\"pred\"]).size()\n"
    "          .reset_index(name=\"count\").sort_values(\"count\", ascending=False))\n"
    "    print(bd.to_string(index=False)); print()\n\n"
    "error_breakdown(errs_v, f\"Vision Batch — {MODEL_DISPLAY}\")\n"
))

# ── Cell 13 — error detail table ──────────────────────────────────────────────
cells.append(code(
    "pd.set_option(\"display.max_colwidth\", 100)\n"
    "if not errs_v.empty:\n"
    "    display(errs_v[[\"sample\",\"page\",\"true\",\"pred\",\"text\"]].reset_index(drop=True))\n"
))

# ── Cell 14 — Section 6 markdown ──────────────────────────────────────────────
cells.append(md(
    "## 6 — Comparison: Qwen2.5-VL 7B (local) vs Claude Opus 4.8 (API)\n\n"
    "Both models use the exact same `vision_batch` strategy and the same page PNG files.  \n"
    "The only difference is the model and provider:\n\n"
    "| | Qwen2.5-VL 7B (M09) | Claude Opus 4.8 (M08) |\n"
    "|---|---|---|\n"
    "| Provider | Ollama (local) | Anthropic API |\n"
    "| Cost | Free | Paid per token |\n"
    "| Image resolution sent | ≤ 900 px (resized) | 150 DPI original |\n"
    "| Context window | 8 192 tokens | Unlimited (API) |\n\n"
    "This comparison isolates the effect of model capability on the vision-batch strategy."
))

# ── Cell 15 — accuracy comparison ────────────────────────────────────────────
cells.append(code(
    "if CLAUDE_AVAILABLE:\n"
    "    # ── Overall accuracy ─────────────────────────────────────────────────\n"
    "    print(\"=== Overall accuracy ===\\n\")\n"
    "    for label, df in [(f\"Qwen2.5-VL 7B  (M09)\", df_v), (\"Claude Opus 4.8 (M08)\", df_c)]:\n"
    "        tc_ = int(df[\"correct\"].sum()); tn_ = int(df[\"total\"].sum())\n"
    "        print(f\"  {label:<24}  {tc_}/{tn_}  ({tc_/tn_*100:.1f}%)\")\n"
    "\n"
    "    # ── Per-sample accuracy side-by-side ──────────────────────────────────\n"
    "    merged = df_v[[\"sample\",\"accuracy\"]].rename(columns={\"accuracy\":\"qwen\"})\\\n"
    "             .merge(df_c[[\"sample\",\"accuracy\"]].rename(columns={\"accuracy\":\"claude\"}), on=\"sample\")\n"
    "    merged[\"delta\"] = merged[\"qwen\"] - merged[\"claude\"]\n"
    "\n"
    "    samples_ordered = merged[\"sample\"].tolist()\n"
    "    x = range(len(samples_ordered))\n"
    "    w = 0.35\n"
    "    fig, ax = plt.subplots(figsize=(13, 5))\n"
    "    bars_q = ax.bar([xi - w/2 for xi in x], merged[\"qwen\"]*100, width=w,\n"
    "                    label=f\"Qwen2.5-VL 7B (M09)\", color=COLOR_VISION, alpha=0.85, edgecolor=\"white\")\n"
    "    bars_c = ax.bar([xi + w/2 for xi in x], merged[\"claude\"]*100, width=w,\n"
    "                    label=\"Claude Opus 4.8 (M08)\", color=COLOR_CLAUDE, alpha=0.85, edgecolor=\"white\")\n"
    "    for bar, v in list(zip(bars_q, merged[\"qwen\"]*100)) + list(zip(bars_c, merged[\"claude\"]*100)):\n"
    "        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,\n"
    "                f\"{v:.0f}%\", ha=\"center\", va=\"bottom\", fontsize=7.5)\n"
    "    ax.axhline(merged[\"qwen\"].mean()*100, color=COLOR_VISION, linestyle=\"--\", linewidth=0.9, alpha=0.7)\n"
    "    ax.axhline(merged[\"claude\"].mean()*100, color=COLOR_CLAUDE, linestyle=\"--\", linewidth=0.9, alpha=0.7)\n"
    "    ax.set_xticks(list(x)); ax.set_xticklabels(samples_ordered, rotation=30)\n"
    "    ax.set_ylim(40, 118); ax.set_ylabel(\"Block-level accuracy (%)\")\n"
    "    ax.set_title(\"Per-sample accuracy — Qwen2.5-VL 7B vs Claude Opus 4.8 — Vision Batch\", pad=10)\n"
    "    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f\"{v:.0f}%\"))\n"
    "    ax.legend(framealpha=0.9); sns.despine(); plt.tight_layout(); plt.show()\n"
    "\n"
    "    # ── Delta table ───────────────────────────────────────────────────────\n"
    "    print(\"\\n=== Per-sample delta (Qwen − Claude) ===\\n\")\n"
    "    print(f\"  {'Sample':<12}  {'Qwen':>8}  {'Claude':>8}  {'Delta':>8}\")\n"
    "    print(\"  \" + \"-\" * 44)\n"
    "    for _, r in merged.iterrows():\n"
    "        sign = '+' if r['delta'] >= 0 else ''\n"
    "        print(f\"  {r['sample']:<12}  {r['qwen']*100:>7.1f}%  {r['claude']*100:>7.1f}%  {sign}{r['delta']*100:>6.1f}%\")\n"
    "    print(\"  \" + \"-\" * 44)\n"
    "    dq = df_v['correct'].sum()/df_v['total'].sum()\n"
    "    dc = df_c['correct'].sum()/df_c['total'].sum()\n"
    "    sign = '+' if (dq-dc) >= 0 else ''\n"
    "    print(f\"  {'OVERALL':<12}  {dq*100:>7.1f}%  {dc*100:>7.1f}%  {sign}{(dq-dc)*100:>6.1f}%\")\n"
    "else:\n"
    "    print(\"Claude vision data not available — run: dmpbridge-experiment experiments/claude-opus-4-8-vision.yaml\")\n"
))

# ── Cell 16 — F1 comparison ───────────────────────────────────────────────────
cells.append(code(
    "if CLAUDE_AVAILABLE:\n"
    "    df_f1_c = compute_f1_rows(conf_c)\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
    "    for ax, df_f1, color, title in [\n"
    "        (axes[0], df_f1_v, COLOR_VISION, f\"Qwen2.5-VL 7B — Vision Batch\"),\n"
    "        (axes[1], df_f1_c, COLOR_CLAUDE, f\"Claude Opus 4.8 — Vision Batch\"),\n"
    "    ]:\n"
    "        xi, bw = range(len(LABELS)), 0.26\n"
    "        ax.bar([i - bw for i in xi], df_f1[\"precision\"]*100, width=bw, label=\"Precision\",\n"
    "               color=color, alpha=0.5, edgecolor=\"white\")\n"
    "        ax.bar([i      for i in xi], df_f1[\"recall\"]*100,    width=bw, label=\"Recall\",\n"
    "               color=color, alpha=0.75, edgecolor=\"white\")\n"
    "        bars = ax.bar([i + bw for i in xi], df_f1[\"f1\"]*100, width=bw, label=\"F1\",\n"
    "                      color=color, edgecolor=\"white\")\n"
    "        for bar, row in zip(bars, df_f1.itertuples()):\n"
    "            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,\n"
    "                    f\"{row.f1*100:.0f}%\", ha=\"center\", va=\"bottom\",\n"
    "                    fontsize=9, fontweight=\"bold\", color=color)\n"
    "        ax.set_xticks(list(xi)); ax.set_xticklabels(SHORT, fontsize=10)\n"
    "        ax.set_ylim(0, 120); ax.set_ylabel(\"Score (%)\")\n"
    "        ax.set_title(title, pad=10)\n"
    "        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f\"{v:.0f}%\"))\n"
    "        ax.legend(loc=\"upper right\", framealpha=0.9); sns.despine(ax=ax)\n"
    "    plt.suptitle(\"F1 per label — Qwen2.5-VL 7B vs Claude Opus 4.8 — Vision Batch\",\n"
    "                 fontsize=13, y=1.02)\n"
    "    plt.tight_layout(); plt.show()\n"
    "\n"
    "    # ── F1 delta table ────────────────────────────────────────────────────\n"
    "    cmp = df_f1_v.set_index(\"label\")[[\"f1\"]].rename(columns={\"f1\":\"qwen_f1\"})\\\n"
    "          .join(df_f1_c.set_index(\"label\")[[\"f1\"]].rename(columns={\"f1\":\"claude_f1\"}))\n"
    "    cmp[\"Δ qwen−claude\"] = cmp[\"qwen_f1\"] - cmp[\"claude_f1\"]\n"
    "    display(cmp.style.format(\"{:.1%}\")\n"
    "            .background_gradient(subset=[\"qwen_f1\",\"claude_f1\"], cmap=\"Blues\"))\n"
))

# ── Cell 17 — confusion matrix comparison ─────────────────────────────────────
cells.append(code(
    "if CLAUDE_AVAILABLE:\n"
    "    mat_c, mat_cn = conf_to_df(conf_c)\n"
    "    fig, axes = plt.subplots(2, 2, figsize=(16, 11))\n"
    "    panels = [\n"
    "        (axes[0,0], mat_v,  \"d\",   CMAP_VISION, \"Qwen2.5-VL 7B — raw counts\"),\n"
    "        (axes[0,1], mat_vn, \".0%\", CMAP_VISION, \"Qwen2.5-VL 7B — row-normalised recall\"),\n"
    "        (axes[1,0], mat_c,  \"d\",   CMAP_CLAUDE, \"Claude Opus 4.8 — raw counts\"),\n"
    "        (axes[1,1], mat_cn, \".0%\", CMAP_CLAUDE, \"Claude Opus 4.8 — row-normalised recall\"),\n"
    "    ]\n"
    "    for ax, mat, fmt, cmap, title in panels:\n"
    "        sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, linewidths=0.6,\n"
    "                    linecolor=\"white\", ax=ax, cbar=False, annot_kws={\"size\":11},\n"
    "                    vmin=(0 if fmt==\".0%\" else None), vmax=(1 if fmt==\".0%\" else None))\n"
    "        ax.set_title(title, pad=8, fontsize=11)\n"
    "        ax.set_xlabel(\"Predicted\", labelpad=6); ax.set_ylabel(\"True\", labelpad=6)\n"
    "    plt.suptitle(\"Confusion matrices — Vision Batch: Qwen2.5-VL 7B vs Claude Opus 4.8\",\n"
    "                 fontsize=13, y=1.01)\n"
    "    plt.tight_layout(); plt.show()\n"
))

# ── Build notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("c:/Users/Nahid/dmpbridge/notebooks/007_qwen2-vl.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Created {out}  ({out.stat().st_size:,} bytes)")
