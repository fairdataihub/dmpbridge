import json
from pathlib import Path

p = Path("c:/Users/Nahid/dmpbridge/notebooks/002_pdfplumber_extraction_eval.ipynb")
raw = p.read_bytes().lstrip(b"\xef\xbb\xbf")
nb = json.loads(raw)

metric_cell = nb["cells"][1]  # keep metric-definition markdown

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


HEADER = md(
    "# PDFPlumber Extraction — Quality Evaluation\n\n"
    "Measures how accurately pdfplumber captures text from DMP PDFs by comparing\n"
    "extracted blocks against manually curated reference text files.\n\n"
    "**Input (extracted):** `data/output/extracted/sampleN.json`  \n"
    "**Input (reference):** `data/input/ground_truth/reference_text/sampleN_reference.txt`  \n"
    "**Output:** per-sample and aggregate quality metrics (no files written)\n"
)

IMPORTS = code(
    "import json\n"
    "import re\n"
    "from collections import Counter\n"
    "from difflib import SequenceMatcher\n"
    "from pathlib import Path\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mticker\n"
    "import pandas as pd\n"
    "import seaborn as sns\n"
    "\n"
    'sns.set_theme(style="ticks")\n'
    'plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",\n'
    '                     "figure.dpi": 110, "axes.titlesize": 12})\n'
    "\n"
    '_ROOT   = Path.cwd() if (Path.cwd() / "dmpbridge").exists() else Path.cwd().parent\n'
    'EXT_DIR = _ROOT / "data/output/extracted"\n'
    'REF_DIR = _ROOT / "data/input/ground_truth/reference_text"\n'
    "\n"
    'print("Extracted blocks :", EXT_DIR, "exists:", EXT_DIR.exists())\n'
    'print("Reference text   :", REF_DIR, "exists:", REF_DIR.exists())\n'
    "\n"
    "\n"
    "def _tokens(text):\n"
    '    return re.sub(r"[^a-z0-9]", " ", text.lower()).split()\n'
    "\n"
    "\n"
    "def evaluate(extracted_text, reference_text):\n"
    "    ext = _tokens(extracted_text)\n"
    "    ref = _tokens(reference_text)\n"
    "    ext_c = Counter(ext)\n"
    "    ref_c = Counter(ref)\n"
    "    correct = sum((ext_c & ref_c).values())\n"
    "    missing = sum(ref_c.values()) - correct\n"
    "    extra   = sum(ext_c.values()) - correct\n"
    "    precision    = correct / (correct + extra)   if (correct + extra)   else 0.0\n"
    "    recall       = correct / (correct + missing) if (correct + missing) else 0.0\n"
    "    f1           = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0\n"
    "    word_capture = correct / sum(ref_c.values()) if ref_c else 0.0\n"
    "    lcs     = sum(b.size for b in SequenceMatcher(None, ext, ref).get_matching_blocks())\n"
    "    rouge_l = lcs / len(ref) if ref else 0.0\n"
    "    return {\n"
    '        "word_capture":    round(word_capture, 4),\n'
    '        "rouge_l":         round(rouge_l, 4),\n'
    '        "precision":       round(precision, 4),\n'
    '        "recall":          round(recall, 4),\n'
    '        "f1":              round(f1, 4),\n'
    '        "extracted_words": len(ext),\n'
    '        "reference_words": len(ref),\n'
    '        "missing_words":   missing,\n'
    '        "extra_words":     extra,\n'
    "    }\n"
)

RUN = code(
    'SCORE_COLS = ["word_capture", "rouge_l", "precision", "recall", "f1"]\n'
    'COUNT_COLS = ["extracted_words", "reference_words", "missing_words", "extra_words"]\n'
    "\n"
    "def _snum(fp): return int(fp.stem.replace('sample', ''))\n"
    "\n"
    "ext_files = sorted(EXT_DIR.glob('sample*.json'), key=_snum)\n"
    "results   = []\n"
    "\n"
    "for ef in ext_files:\n"
    "    stem = ef.stem\n"
    "    rf   = REF_DIR / f'{stem}_reference.txt'\n"
    "    if not rf.exists():\n"
    "        print(f'  missing reference: {rf.name}')\n"
    "        continue\n"
    "    blocks   = json.loads(ef.read_text(encoding='utf-8'))\n"
    "    ext_text = ' '.join(b['text'] for b in blocks)\n"
    "    ref_text = rf.read_text(encoding='utf-8')\n"
    "    m = evaluate(ext_text, ref_text)\n"
    "    m['sample'] = stem\n"
    "    results.append(m)\n"
    "    print(f\"  {stem}: F1={m['f1']:.3f}  capture={m['word_capture']:.3f}  \"\n"
    "          f\"ROUGE-L={m['rouge_l']:.3f}  missing={m['missing_words']}  extra={m['extra_words']}\")\n"
    "\n"
    "df = pd.DataFrame(results).set_index('sample')\n"
    "print()\n"
    "display(\n"
    "    df[SCORE_COLS + COUNT_COLS]\n"
    "    .style.format({c: '{:.3f}' for c in SCORE_COLS})\n"
    "    .background_gradient(subset=SCORE_COLS, cmap='RdYlGn', vmin=0.8, vmax=1.0)\n"
    ")\n"
)

CHART1 = code(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    'colors = ["#6366f1", "#7c3aed", "#0369a1", "#059669", "#dc2626"]\n'
    "w = 0.15\n"
    "x = range(len(df))\n"
    "\n"
    "for j, (col, color) in enumerate(zip(SCORE_COLS, colors)):\n"
    "    axes[0].bar([xi + j * w for xi in x], df[col] * 100,\n"
    "                width=w, label=col, color=color, edgecolor='white', alpha=0.88)\n"
    "axes[0].set_xticks([xi + w * 2 for xi in x])\n"
    "axes[0].set_xticklabels(df.index, rotation=30, ha='right')\n"
    "axes[0].set_ylim(50, 105)\n"
    "axes[0].set_ylabel('Score (%)')\n"
    "axes[0].set_title('Quality scores per sample')\n"
    "axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))\n"
    "axes[0].legend(fontsize=8, ncol=2)\n"
    "sns.despine(ax=axes[0])\n"
    "\n"
    "means = df[SCORE_COLS].mean() * 100\n"
    "bars = axes[1].bar(means.index, means, color=colors, edgecolor='white', alpha=0.88)\n"
    "for bar, v in zip(bars, means):\n"
    "    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,\n"
    "                 f'{v:.1f}%', ha='center', va='bottom', fontsize=9)\n"
    "axes[1].set_ylim(50, 105)\n"
    "axes[1].set_ylabel('Mean score (%)')\n"
    "axes[1].set_title('Mean scores across all 10 samples')\n"
    "axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))\n"
    "sns.despine(ax=axes[1])\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)

CHART2 = code(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "x    = range(len(df))\n"
    "lbls = df.index.tolist()\n"
    "\n"
    "axes[0].bar([xi - 0.2 for xi in x], df['reference_words'], 0.35,\n"
    "            label='Reference', color='#6366f1', edgecolor='white', alpha=0.88)\n"
    "axes[0].bar([xi + 0.2 for xi in x], df['extracted_words'], 0.35,\n"
    "            label='Extracted', color='#059669', edgecolor='white', alpha=0.88)\n"
    "axes[0].set_xticks(x); axes[0].set_xticklabels(lbls, rotation=30, ha='right')\n"
    "axes[0].set_ylabel('Word count')\n"
    "axes[0].set_title('Extracted vs reference word counts')\n"
    "axes[0].legend(fontsize=9)\n"
    "sns.despine(ax=axes[0])\n"
    "\n"
    "axes[1].bar([xi - 0.2 for xi in x], df['missing_words'], 0.35,\n"
    "            label='Missing', color='#dc2626', edgecolor='white', alpha=0.88)\n"
    "axes[1].bar([xi + 0.2 for xi in x], df['extra_words'],   0.35,\n"
    "            label='Extra',   color='#f59e0b', edgecolor='white', alpha=0.88)\n"
    "axes[1].set_xticks(x); axes[1].set_xticklabels(lbls, rotation=30, ha='right')\n"
    "axes[1].set_ylabel('Word count')\n"
    "axes[1].set_title('Missing and extra words per sample')\n"
    "axes[1].legend(fontsize=9)\n"
    "sns.despine(ax=axes[1])\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)

SUMMARY = code(
    "agg = df[SCORE_COLS].agg(['mean', 'min', 'max'])\n"
    "print('Aggregate quality scores across all 10 samples:\\n')\n"
    "print(f\"  {'Metric':<16} {'Mean':>8} {'Min':>8} {'Max':>8}\")\n"
    "print('  ' + '-' * 44)\n"
    "for col in SCORE_COLS:\n"
    "    print(f\"  {col:<16} {agg.loc['mean', col]:>7.1%}  {agg.loc['min', col]:>7.1%}  {agg.loc['max', col]:>7.1%}\")\n"
    "\n"
    "total_ref  = df['reference_words'].sum()\n"
    "total_ext  = df['extracted_words'].sum()\n"
    "total_miss = df['missing_words'].sum()\n"
    "total_xtra = df['extra_words'].sum()\n"
    "print(f'\\n  Total reference words : {total_ref:,}')\n"
    "print(f'  Total extracted words : {total_ext:,}')\n"
    "print(f'  Total missing words   : {total_miss:,}  ({total_miss/total_ref:.1%} of reference)')\n"
    "print(f'  Total extra words     : {total_xtra:,}  ({total_xtra/total_ext:.1%} of extracted)')\n"
)

nb["cells"] = [
    HEADER,
    metric_cell,
    md("## 1 — Imports and metric functions"),
    IMPORTS,
    md("## 2 — Run evaluation on all 10 samples"),
    RUN,
    md("## 3 — Score overview"),
    CHART1,
    md("## 4 — Word count comparison"),
    CHART2,
    md("## 5 — Aggregate summary"),
    SUMMARY,
]

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {len(nb['cells'])} cells")
