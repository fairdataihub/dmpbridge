"""Build notebooks/export-labeled-results-to-excel.ipynb.

Review export for the ``gemma4-e4b_pdfplumber_whole_doc`` pipeline: the FINAL
structured JSON (stage 4 — what a user of the package receives, after
structuring and the annotation rules) flattened into a clean two-column
(label, text) table per sample, and written to
``data/output/pdfplumber_fallback_lightonocr_and_gemma.xlsx`` — one sheet per
sample plus a combined sheet with a ``sample`` column.

An earlier version exported stage 2 (the model's raw labeled blocks); that
disagreed with the final results users see — sample 13's final JSON has 10
sections and 5 questions, while its raw output was 25 blocks — so the export
now follows the final JSON.

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
        "# Export final results to Excel — gemma4:e4b + pdfplumber",
        "",
        f"The pipeline's **final output** for every document under the `{TAG}` tag lives in",
        "`data/output/4_final/` — the structured DMP JSON after the annotation rules, the",
        "file a user of the package receives. This notebook flattens each one into a",
        "two-column **(label, text)** table in document order and writes them all to one",
        "Excel workbook:",
        "",
        f"- `{XLSX}`",
        "- one sheet per sample",
        "",
        "Because this reads stage 4, every sheet matches the final JSON exactly —",
        "including what the converter merged and what the rules filled in. (The model's",
        "raw pre-structure blocks live in `2_labeled/` if those are ever needed instead.)",
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
        "FINAL_DIR = Path('data/output/4_final') / TAG",
        f"XLSX = Path({XLSX!r})",
        "",
        "pd.set_option('display.max_colwidth', 90)",
    ]),

    md("md-1", [
        "## 1. The flattening function",
        "",
        "One function, one job: a final structured JSON in, a tidy `(label, text)` table",
        "out, rows in document order — the document title first, then each section's",
        "title, description, questions and answers. Empty fields produce no row, so the",
        "table holds exactly what the final JSON holds.",
    ]),
    code("parse", [
        "def flatten_final_json(path: Path) -> pd.DataFrame:",
        "    \"\"\"Flatten one final structured DMP JSON into a (label, text) table.\"\"\"",
        "    data = json.loads(path.read_text(encoding='utf-8'))",
        "    template = data.get('narrative', {}).get('template')",
        "    if template is None:",
        "        raise ValueError(f'{path.name}: not a structured DMP JSON (no narrative.template)')",
        "    rows = []",
        "",
        "    def add(label, text):",
        "        if text and text.strip():",
        "            rows.append({'label': label, 'text': text.strip()})",
        "",
        "    add('title', template.get('title', ''))",
        "    for section in template.get('section', []):",
        "        add('section.title', section.get('title', ''))",
        "        add('section.description', section.get('description', ''))",
        "        for q in section.get('question', []):",
        "            add('question.text', q.get('text', ''))",
        "            add('answer.text', q.get('answer', {}).get('json', {}).get('answer', ''))",
        "    return pd.DataFrame(rows, columns=['label', 'text'])",
    ]),

    md("md-2", [
        "## 2. Flatten every sample under the tag",
        "",
        "The summary shows, per document, how many rows the final JSON yields and how",
        "they split across the five labels.",
    ]),
    code("load", [
        "files = sorted(FINAL_DIR.glob('sample*.json'),",
        "               key=lambda p: int(p.stem.replace('sample', '')))",
        "tables = {p.stem: flatten_final_json(p) for p in files}",
        "",
        "summary = pd.DataFrame([",
        "    {'sample': name, 'rows': len(df), **df['label'].value_counts().to_dict()}",
        "    for name, df in tables.items()",
        "]).fillna(0)",
        "for col in summary.columns:",
        "    if col not in ('sample',):",
        "        summary[col] = summary[col].astype(int)",
        "display(summary.set_index('sample'))",
    ]),
    md("md-2b", [
        "One table up close — the format every sheet in the workbook uses.",
    ]),
    code("peek", [
        "display(tables['sample13'].head(10))",
    ]),

    md("md-3", [
        "## 3. Export to Excel",
        "",
        "One sheet per sample, two columns each.",
    ]),
    code("export", [
        "XLSX.parent.mkdir(parents=True, exist_ok=True)",
        "",
        "with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:",
        "    for name, df in tables.items():",
        "        df.to_excel(writer, sheet_name=name, index=False)",
        "    for sheet in writer.sheets.values():",
        "        sheet.column_dimensions['A'].width = 22",
        "        sheet.column_dimensions['B'].width = 130",
        "",
        "print(f'wrote {XLSX}  ({XLSX.stat().st_size / 1024:.0f} KB)')",
        "print(f'{len(tables)} sample sheets, {sum(len(df) for df in tables.values())} rows total')",
    ]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
