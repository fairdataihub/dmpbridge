# Build scripts

These regenerate the report, the diagram and the notebooks. Everything they produce is a
build artefact — if one is lost or looks wrong, re-run the script rather than editing the
output by hand, or the two will drift apart.

Run them from the repository root.

## The report

```bash
python scripts/build/build_report.py
```

Writes `Report-doc/project_report.docx`. All prose, tables and section numbering live in
this script; the `.docx` is not tracked in git because it is binary and reproducible.

Embeds `Report-doc/pipeline_diagram.png` as Figure 1, so build the diagram first if it has
changed.

## The pipeline diagram

```bash
python scripts/build/make_pipeline_diagram.py
```

Writes `Report-doc/pipeline_diagram.png` — the version used in the Word report.

There is a second diagram of the same pipeline in `docs/pipeline.md`, written in Mermaid for
GitHub. **The two are maintained separately: changing one means changing the other.**
`docs/DIAGRAMS.md` explains how to merge them into a single source if Node.js is ever
installed.

## The notebooks

```bash
# one per model
python scripts/build/rebuild_results_notebook.py \
    notebooks/1-llama31-8b-results-pdfplumber.ipynb "llama3.1:8b" llama3.1-8b_pdfplumber_whole_doc

python scripts/build/rebuild_rules_notebook.py     # annotation_conversion_test.ipynb
```

These **replace every cell**, so any manual edit to a notebook is lost on the next run. That
is deliberate — the three results notebooks are meant to stay identical in structure so they
can be compared side by side. Change the script, not the notebook.

After rebuilding, execute them to populate the outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/1-llama31-8b-results-pdfplumber.ipynb
```

## Why these are tracked

They were originally written in a temporary working folder. Since the `.docx` is not in git
either, losing that folder would have meant losing both the report and any way to rebuild
it. Keeping the generators in the repository makes the artefacts reproducible from source.
