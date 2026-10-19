# Scoring

[← Main README](../README.md) · [All docs](README.md)

How output is compared against hand-annotated reference documents, and how to read the
numbers.

## The short version

```bash
dmpbridge-evaluate      gemma4-e4b_pdfplumber_whole_doc    # Path A
dmpbridge-evaluate-new  gemma4-e4b_pdfplumber_whole_doc    # Path B
```

Three things to know before you read any number:

1. **There are always two scores, A and B.** Report both — they use different reference
   versions, so the gap between them is the annotation rules' contribution, not a second
   opinion on one number.
2. **Differences below ~0.005 F1 are noise.** The measured floor is ±0.002.
3. **You don't have to compute anything.** The results notebooks already have every number —
   see [Where the numbers live](#where-the-numbers-live).

---

## Two paths, always both

The annotation standard changed partway through the project, so everything is scored twice:

- **Path A** — the structured output (stage 3) against the **original** annotation
- **Path B** — the final output (stage 4, after filling blank questions using
  `data/input/Rules.xlsx`) against the **newer** annotation

**Always report both.** They use different reference versions, so the gap between them
measures the annotation rules' contribution — it is not a second opinion on one number.
Support differs between the paths for the same reason, so compare scores, not counts.

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

## The RDA JSON

```bash
dmpbridge-evaluate-rda llama3.1-8b_pdfplumber_rda          # a folder in data/output/5_rda/
```

`dmpbridge-pipeline` runs this automatically at the end (`--no-evaluate` skips it), for
every sample that has a hand-made reference. It prints a line per sample and saves, in
`5_rda/<rda-tag>/evaluation/`, the field matrix as `matrix_sampleN.png` and every score in
`evaluation_results.xlsx`.

The rules are the ones the RDA prompt experiments use — the same code, in
`dmpbridge/rda/evaluate.py` — so a pipeline run and a notebook run of the same output give
the same numbers:

| What | Where |
|---|---|
| hand-made references, samples 1–10 and 14 | `data/input/rda_ground_truth/RDA_DMP_sample<n>_manual_annotation.json` |
| each prompt version's results, prompt and scores | `notebooks/rda-experiments/runs/<version>/` |
| run a prompt version and score it | `notebooks/pdf-to-rda-dmp-json.ipynb` |
| all models and samples for one version | `notebooks/rda-evaluation-overview.ipynb` |
| field by field for one sample | `notebooks/rda-field-matrix.ipynb` |

Text fields match with the same 75% word-containment rule as above. Two sets of labels are
used:

- **The evaluation** (`pdf-to-rda-dmp-json.ipynb`, the overview) scores every field either
  side wrote. **Correct** — the model's value matches the reference. **Hallucinated** — the
  model wrote a wrong value, a value the reference doesn't have, or left blank a field the
  reference fills in. **Missed** — the reference has it and the model never wrote the field.
  Precision, recall and F1 follow from those counts.
- **The field matrix** (`rda-field-matrix.ipynb`) looks only at fields the reference fills
  in. **Correct** — the model's value matches. **Mismatch** — the model wrote a different
  value. **Missed** — the model wrote nothing there.

Judge a prompt change on all of samples 1–10, never on one sample — the first five prompt
versions were tuned on sample 14 alone and did not carry over to the others.

---

## The annotation rules — `data/input/Rules.xlsx`

**The column order is load-bearing.** It has changed once already, silently remapping every
row and dropping agreement with the reference files from 10/10 to 2/10 — without the action
text changing, so nothing looked broken.

`RULE_FIELDS` in `dmpbridge/evaluation/annotation_rules.py` records the expected order, and
a test reads the sheet header at test time and fails on any mismatch. **If that test fails,
the sheet was re-ordered — fix `RULE_FIELDS`, do not edit the test.**
