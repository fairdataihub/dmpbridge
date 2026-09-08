"""Build notebooks/export-labeled-results-to-excel.ipynb.

A small review notebook: parse the model's raw labeled JSON for every sample
under the ``gemma4-e4b_pdfplumber_whole_doc`` tag into a clean two-column
(label, text) table, show it, and export everything to
``output_folder/pdfplumber_fallback_lightonocr_and_gemma.xlsx`` — one sheet
per sample plus a combined sheet with a ``sample`` column for filtering.

    python scripts/build/build_excel_export_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/export-labeled-results-to-excel.ipynb
"""
import json
from pathlib import Path

NB = Path("notebooks/export-labeled-results-to-excel.ipynb")
TAG = "gemma4-e4b_pdfplumber_whole_doc"
XLSX = "data/output/pdfplumber_fallback_lightonocr_and_gemma.xlsx"


def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = [
    md("title", [
        "# Export labeled results to Excel — gemma4:e4b + pdfplumber",
        "",
        f"The model's raw output for every document under the `{TAG}` tag lives in",
        "`data/output/2_labeled/` as JSON: a flat array of `{\"text\", \"label\"}` items.",
        "This notebook parses each file into a clean two-column **(label, text)** table",
        "for review, and writes them all to one Excel workbook:",
        "",
        f"- `{XLSX}`",
        "- one sheet per sample, plus a `all_samples` sheet with a `sample` column",
        "",
        "Note on the filename: the tag can include documents rescued by the LightOnOCR",
        "fallback (a broken text layer is re-read from page images; the log records it).",
        "Sample 11 in its current form is a clean PDF and needed no fallback.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import json",
        "",
        "import pandas as pd",
        "from IPython.display import display",
        "",
        f"TAG = {TAG!r}",
        "LABELED_DIR = Path('data/output/2_labeled') / TAG",
        f"XLSX = Path({XLSX!r})",
        "",
        "pd.set_option('display.max_colwidth', 90)",
    ]),

    md("md-1", [
        "## 1. The parsing function",
        "",
        "One function, one job: raw JSON file in, tidy `(label, text)` DataFrame out.",
        "It validates as it parses — a malformed file or an unknown label should be",
        "seen at review time, not silently passed through.",
    ]),
    code("parse", [
        "KNOWN_LABELS = ('title', 'section.title', 'section.description',",
        "                'question.text', 'answer.text')",
        "",
        "",
        "def parse_labeled_json(path: Path) -> pd.DataFrame:",
        "    \"\"\"Parse one raw model-output JSON into a two-column (label, text) table.",
        "",
        "    Raises ValueError if the file is not the expected flat array of",
        "    {'text', 'label'} objects; prints a notice for any label outside the",
        "    five known ones rather than dropping the row.",
        "    \"\"\"",
        "    raw = json.loads(path.read_text(encoding='utf-8'))",
        "    if not isinstance(raw, list):",
        "        raise ValueError(f'{path.name}: expected a JSON array, got {type(raw).__name__}')",
        "    rows = []",
        "    for i, item in enumerate(raw):",
        "        if not isinstance(item, dict) or 'text' not in item or 'label' not in item:",
        "            raise ValueError(f'{path.name}[{i}]: expected {{text, label}}, got {item!r:.60}')",
        "        if item['label'] not in KNOWN_LABELS:",
        "            print(f'  note: {path.name}[{i}] has unknown label {item[\"label\"]!r}')",
        "        rows.append({'label': item['label'], 'text': item['text'].strip()})",
        "    return pd.DataFrame(rows, columns=['label', 'text'])",
    ]),

    md("md-2", [
        "## 2. Parse every sample under the tag",
    ]),
    code("load", [
        "files = sorted(LABELED_DIR.glob('sample*.json'),",
        "               key=lambda p: int(p.stem.replace('sample', '')))",
        "tables = {p.stem: parse_labeled_json(p) for p in files}",
        "",
        "summary = pd.DataFrame([",
        "    {'sample': name, 'blocks': len(df), **df['label'].value_counts().to_dict()}",
        "    for name, df in tables.items()",
        "]).fillna(0).astype({l: int for l in KNOWN_LABELS if any(True for _ in tables)}, errors='ignore')",
        "display(summary.set_index('sample'))",
    ]),
    md("md-2b", [
        "One table up close — the format every sheet in the workbook uses.",
    ]),
    code("peek", [
        "display(tables['sample1'].head(8))",
    ]),

    md("md-3", [
        "## 3. Export to Excel",
        "",
        "One sheet per sample (two columns), plus `all_samples` with a `sample` column",
        "so the whole tag can be filtered in one view. Column widths are set so the",
        "text is readable without resizing.",
    ]),
    code("export", [
        "XLSX.parent.mkdir(parents=True, exist_ok=True)",
        "",
        "with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:",
        "    combined = pd.concat(",
        "        [df.assign(sample=name)[['sample', 'label', 'text']] for name, df in tables.items()],",
        "        ignore_index=True)",
        "    combined.to_excel(writer, sheet_name='all_samples', index=False)",
        "    for name, df in tables.items():",
        "        df.to_excel(writer, sheet_name=name, index=False)",
        "    for sheet in writer.sheets.values():",
        "        widths = {'A': 14, 'B': 22, 'C': 120} if sheet.title == 'all_samples' else {'A': 22, 'B': 130}",
        "        for col, w in widths.items():",
        "            sheet.column_dimensions[col].width = w",
        "",
        "print(f'wrote {XLSX}  ({XLSX.stat().st_size / 1024:.0f} KB)')",
        "print(f'{len(tables)} sample sheets + all_samples ({len(combined)} rows total)')",
    ]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
