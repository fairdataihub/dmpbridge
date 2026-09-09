# Scoring

How output is compared against hand-annotated reference documents, and how to read the
numbers.

---

## Two paths, always both

The annotation standard changed partway through the project, so everything is scored twice:

- **Path A** — the structured output (stage 3) against the **original** annotation
- **Path B** — the final output (stage 4, after filling blank questions using
  `data/input/Rules.xlsx`) against the **newer** annotation

**Always report both.** They use different reference versions, so the gap between them
measures the annotation rules' contribution — it is not a second opinion on one number.
Support differs between the paths for the same reason, so compare scores, not counts.

```bash
dmpbridge-evaluate      gemma4-e4b_pdfplumber_whole_doc    # Path A
dmpbridge-evaluate-new  gemma4-e4b_pdfplumber_whole_doc    # Path B
```

Both paths are also driven from one YAML file, which is what the results notebooks are
built from:

```bash
dmpbridge-experiment experiments/llama3.1-8b-wholedoc.yaml --evaluate
```

A new annotation source ("Path C") doesn't need new code — it's a new entry in the
`evaluation:` list in the YAML. `experiments/full-example.yaml` shows every field, and
`EvaluationPath` in `dmpbridge/evaluation/evaluate.py` is the class behind it.

---

## The match rule

A predicted item matches a reference item when enough of its words are contained in it —
**75% by default**, so partial credit is allowed. Pass `threshold=1.0` to require an exact
match instead. Precision, recall and F1 follow as usual.

`notebooks/analysis-threshold-comparison.ipynb` shows the same runs at both thresholds.

---

## Reading the numbers

**The noise floor is ±0.002 F1 — one text block.** Measured by running one configuration
three times: every count identical, F1 unmoved. `temperature: 0.0` is genuinely
deterministic here, so **differences above about 0.005 are real signal** and anything
smaller is not meaningful. Earlier notes claiming a few points of run-to-run variation were
assumptions, never measurements.

`python scripts/noise_floor.py` re-runs that check.

Two things that do move results, and are easy to do by accident:

- **The prompt is whitespace-significant.** Moving three blank lines in `SYSTEM_PROMPT`
  (`dmpbridge/prompts/constants.py`), with no wording change at all, moved F1 by 1.9 points and
  false positives by 13 on `llama3.1:8b`. Both layouts reproduced to within one block, so
  this is not noise. An editor that strips or adds a trailing newline on save can change
  results more than most rewording.
- **Prompt changes do not transfer between models.** One edit gained 0.004 on `llama3.1:8b`
  while costing 0.040 on `gemma4:e4b` and 0.006 on `llama3.3:70b`. Re-run **every** model
  after any prompt change — a partial re-run produces figures that cannot be compared.

---

## Where the numbers live

Results change often enough that they are not hardcoded in the README. All four models are
fully evaluated under the current pipeline (`pdfplumber`, whole-document, both paths):

| Notebook | What it shows |
|---|---|
| `notebooks/results-<model>-pdfplumber.ipynb` | one model in full — both paths, per class, per document, confusion matrices |
| `notebooks/comparison-4models-pdfplumber-75pct-overlap.ipynb` | all four models side by side |
| `notebooks/comparison-gemma-extractors.ipynb` | all three extractors, one model |
| `notebooks/analysis-confidence-scores.ipynb` | per-label confidence |

**The results notebooks are generated.** Don't hand-edit them — edit the builder in
`scripts/build/` and re-run, then execute with
`jupyter nbconvert --to notebook --execute --inplace`.

Research code: results are provisional and the evaluation set is small.

---

## The annotation rules — `data/input/Rules.xlsx`

**The column order is load-bearing.** It has changed once already, silently remapping every
row and dropping agreement with the reference files from 10/10 to 2/10 — without the action
text changing, so nothing looked broken.

`RULE_FIELDS` in `dmpbridge/evaluation/annotation_rules.py` records the expected order, and
a test reads the sheet header at test time and fails on any mismatch. **If that test fails,
the sheet was re-ordered — fix `RULE_FIELDS`, do not edit the test.**
