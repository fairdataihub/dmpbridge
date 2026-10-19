# DMPBridge

[![Contributors](https://img.shields.io/github/contributors/fairdataihub/dmpbridge?style=flat-square&logo=github&logoColor=white&color=2ea44f)](https://github.com/fairdataihub/dmpbridge/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/fairdataihub/dmpbridge?style=flat-square&logo=github&logoColor=white&color=f9d949)](https://github.com/fairdataihub/dmpbridge/stargazers)
[![Issues](https://img.shields.io/github/issues/fairdataihub/dmpbridge?style=flat-square&logo=github&logoColor=white&color=ff7a00)](https://github.com/fairdataihub/dmpbridge/issues)
[![License](https://img.shields.io/github/license/fairdataihub/dmpbridge?style=flat-square&color=1f6feb)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-9e9e9e?style=flat-square)](#how-to-cite)


A Data Management Plan (DMP) describes how researchers will manage, preserve, and share
project data in line with the Findable, Accessible, Interoperable, and Reusable (FAIR)
principles and other relevant disciplinary guidelines. DMPs are commonly required by funders
with every grant proposal, but the formats and requirements imposed by each funder keep
evolving, and no two funders' PDFs are structured the same way.

DMPBridge is an open-source (MIT License), Python-based pipeline that converts DMP PDFs from
any funder format into DMP Tool JSON, combining a narrative portion that mirrors DMP Tool's
internal structure with the RDA DMP Common Standard JSON for machine-actionable output.

This project is part of a broader extension of the DMP Tool platform. The ultimate goal is
to integrate the DMP Chef pipeline into the [DMP Tool](https://dmptool.org/), providing
researchers with a familiar and convenient user interface that does not require any coding
knowledge.

👉 Learn more: **[DMP Bridge](https://fairdataihub.org/dmp-bridge)**

> **This is an active research project, things change often.**

---
## Standards followed
The overall codebase is organized in alignment with the **[FAIR-BioRS guidelines](https://fair-biors.org/)**. All Python code follows **[PEP 8](https://peps.python.org/pep-0008/)** conventions, including consistent formatting, inline comments, and docstrings.Project dependencies are fully captured in that same
**[pyproject.toml](https://github.com/fairdataihub/dmpbridge/blob/main/pyproject.toml)**.

---

## How it works

```mermaid
flowchart TD
    PDF["<b>DMP PDF</b>"]

    PDF --> READ["<b>Read the PDF</b><br/><small>pdfplumber — text, fonts, underlines</small>"]
    READ --> CHECK{"<b>Readable text?</b>"}
    CHECK -- yes --> S1["<b>1. Text blocks</b>"]
    CHECK -- "no — fall back" --> OCR["<b>Read page images</b><br/><small>LightOnOCR</small>"]
    OCR --> S1
    S1 --> LABEL["<b>Label each block</b><br/><small>llama3.1:8b · gemma4:e4b · llama3.3:70b · qwen2.5:14b</small>"]
    LABEL --> S2["<b>2. Labeled blocks</b>"]
    S2 --> BUILD["<b>Build the structure</b>"]
    BUILD --> S3["<b>3. Structured JSON</b>"]

    S3 --> PATHA["<b>Path A</b><br/>score vs old annotation"]
    S3 --> RULES["<b>Apply the rules</b><br/><small>Rules.xlsx</small>"]
    RULES --> S4["<b>4. Final JSON</b>"]
    S4 --> PATHB["<b>Path B</b><br/>score vs new annotation"]

    S1 --> RDA["<b>Extract the RDA JSON</b><br/><small>same text · RDA prompt · maDMP schema</small>"]
    RDA --> S5["<b>5. RDA JSON</b>"]
    S5 --> MERGE["<b>Merge</b>"]
    S4 --> MERGE
    MERGE --> S6["<b>6. Merged JSON</b><br/><small>RDA + narrative</small>"]

    classDef input  fill:#1E406E,stroke:#1E406E,color:#ffffff
    classDef step   fill:#F4F7FB,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef cached fill:#ffffff,stroke:#0F766E,stroke-width:2px,color:#111
    classDef data   fill:#ffffff,stroke:#1E406E,stroke-width:2px,color:#111
    classDef rules  fill:#ffffff,stroke:#B45309,stroke-width:2px,color:#111
    classDef pathA  fill:#EDF3FA,stroke:#3C6FA8,stroke-width:2px,color:#111
    classDef pathB  fill:#FDF4E9,stroke:#B45309,stroke-width:2px,color:#111
    classDef check  fill:#FDF4E9,stroke:#B45309,stroke-width:1px,color:#111

    class PDF input
    class READ,LABEL,BUILD,RULES,OCR,RDA,MERGE step
    class CHECK check
    class S1 cached
    class S2,S3,S5,S6 data
    class S4 rules
    class PATHA pathA
    class PATHB pathB
```

The model's one job is to decide what each piece of text *is*. Every block gets one of five
labels, and the structure is built from those:

| Label | Meaning |
|---|---|
| `title` | Document title |
| `section.title` | Section heading |
| `section.description` | Instructions written by the funder |
| `question.text` | A question or prompt |
| `answer.text` | The researcher's response |

Alongside the narrative, a second model reads the same text and fills in the
[RDA DMP Common Standard](https://github.com/RDA-DMP-Common/RDA-DMP-Common-Standard)
(maDMP 1.2) JSON — title, contact, datasets, and so on. The two are merged into one file.

---

## Quick start

Three commands' worth of setup, then one command to run it. You need
[Python 3.10+](https://www.python.org/downloads/) and [Ollama](https://ollama.com). 
```bash
# 1 — get the code and install it
git clone https://github.com/fairdataihub/dmpbridge.git
cd dmpbridge
python -m venv venv
venv\Scripts\Activate.ps1          # macOS / Linux:  source venv/bin/activate
pip install -e .

# 2 — download a local model (~3 GB, one time)
ollama pull gemma4:e4b

# 3 — label your first document
dmpbridge-wholedoc --model gemma4:e4b --fallback auto --start 1 --end 1
```

That takes about 20 seconds and writes your result to:

```
data/output/4_final/gemma4-e4b_pdfplumber_whole_doc/sample1.json
```

Change `--end 1` to `--end 13` to run the whole sample set. A document that needs the OCR
rescue below takes longer — around 40 seconds.

### What you get

The PDF goes in as flat text. What comes out is nested — sections, the questions inside
them, and the researcher's answer attached to each question:

```json
{
  "title": "DATA MANAGEMENT AND SHARING PLAN",
  "section": [
    {
      "title": "Element 1: Data Type:",
      "order": 1,
      "question": [
        {
          "text": "A. Types and amount of scientific data expected to be generated…",
          "answer": {
            "json": {
              "type": "textArea",
              "answer": "This secondary data analysis project will analyze deidentified…"
            }
          }
        }
      ]
    }
  ]
}
```

### Add the RDA JSON

One command runs all six stages — the narrative part above, then the RDA part:

```bash
ollama pull llama3.1:8b
dmpbridge-pipeline --narrative-model gemma4:e4b --rda-model llama3.1:8b --start 1 --end 1
```

The merged result is written to:

```
data/output/6_merged/sample1.json
```

with the RDA JSON first and the narrative JSON after it:

```json
{
  "dmp":       { "title": "…", "contact": { … }, "dataset": [ … ] },
  "narrative": { "template": { "title": "…", "section": [ … ] } }
}
```

Already ran the narrative part? `dmpbridge-rda --model llama3.1:8b --narrative-model gemma4:e4b`
adds just the RDA part.

For a sample with a hand-made reference, the pipeline then scores the RDA JSON and saves a
field matrix — every annotated field marked Correct, Mismatch or Missed — in
`data/output/5_rda/evaluation/`. To score again later: `dmpbridge-evaluate-rda`.

### What the RDA part fills in

The model fills in this form, following the
[RDA DMP Common Standard](https://github.com/RDA-DMP-Common/RDA-DMP-Common-Standard).
A `[REQUIRED]` field is always written — with the plan's value, otherwise its default;
a `[RECOMMENDED]` field only when the plan gives that information.

```json
{
  "dmp": {
    "contact": {
      "contact_id": [{
        "identifier": "[REQUIRED] ORCID preferred but email otherwise",
        "type": "[REQUIRED]"
      }],
      "mbox": "[REQUIRED]",
      "name": "[REQUIRED]"
    },
    "created": "[REQUIRED]",
    "dataset": [{
      "dataset_id": {
        "identifier": "[REQUIRED]",
        "type": "[REQUIRED]"
      },
      "description": "[RECOMMENDED]",
      "distribution": [{
        "byte_size": "[RECOMMENDED]",
        "data_access": "[REQUIRED] default to 'closed'",
        "description": "[RECOMMENDED]",
        "host": {
          "host_id": [{
            "identifier": "[REQUIRED] if host info is present, ROR id preferred",
            "type": "[REQUIRED] if host info is present"
          }],
          "title": "[REQUIRED] if host info is present"
        },
        "issued": "[RECOMMENDED]",
        "license": [{
          "license_ref": "[REQUIRED] if license is present",
          "start_date": "[REQUIRED] if license is present, default to current date time"
        }],
        "title": "[REQUIRED]"
      }],
      "metadata": [{
        "description": "[RECOMMENDED]",
        "language": "[REQUIRED] default to 'eng'",
        "metadata_standard_id": [
          {
            "identifier": "[REQUIRED] if metadata standard info is present",
            "type": "[REQUIRED] if metadata standard info is present"
          }
        ]
      }],
      "personal_data": "[REQUIRED] default to 'unknown'",
      "sensitive_data": "[REQUIRED] default to 'unknown'",
      "title": "[REQUIRED]",
      "type": "[REQUIRED] default to 'dataset'"
    }],
    "description": "[RECOMMENDED]",
    "dmp_id": {
      "identifier": "[REQUIRED]",
      "type": "[REQUIRED]"
    },
    "ethical_issues_exist": "[REQUIRED] default to 'unknown'",
    "language": "[REQUIRED] default to 'eng'",
    "modified": "[REQUIRED]",
    "project": [{
      "description": "[REQUIRED]",
      "end": "[RECOMMENDED]",
      "funding": [{
        "funder_id": {
          "identifier": "[REQUIRED] ROR id preferred",
          "type": "[REQUIRED]"
        },
        "funding_status": "[REQUIRED] default to 'planned'",
        "grant_id": {
          "identifier": "[RECOMMENDED]",
          "type": "[RECOMMENDED]"
        }
      }],
      "start": "[RECOMMENDED]",
      "title": "[REQUIRED]"
    }],
    "title": "[REQUIRED]"
  }
}
```

---

## Ways to run it

### Option A — Jupyter demo

See each stage happen, with the settings and the finished document side by side.

1. Open [`demo/config.yaml`](demo/config.yaml) and set the models:

   ```yaml
   narrative:
     model: gemma4:e4b    # labels the text, builds the DMP Tool JSON
   rda:
     model: llama3.1:8b   # fills in the RDA JSON; delete these two lines to skip it
   sample_start: 1
   sample_end: 1        # which sample PDFs to run
   ```

2. Open [`notebooks/demo-from-yaml-config.ipynb`](notebooks/demo-from-yaml-config.ipynb)
   and click **Run All**. Nothing inside the notebook needs editing.

3. The notebook prints the finished document — title, sections, questions, answers — and
   the RDA JSON.

No Jupyter? The same config runs from the terminal, writing each stage into `demo/output/`:

```bash
python scripts/run_demo.py
```

### Option B — one command

Convert a PDF and get the JSON. Nothing to configure.

1. Put your PDF anywhere you like.

2. Run:

   ```bash
   dmpbridge my-plan.pdf --model gemma4:e4b
   ```

3. Open **`my-plan_labeled_structured.json`**, written next to your PDF. That's the DMP Tool
   JSON. (`my-plan_labeled.json` is also written — the flat list of labeled blocks behind
   it.)

---

## Where the output goes

```
data/output/
├── 1_extracted/<extractor>/sampleN.json     text blocks, no labels
├── 2_labeled/<tag>/sampleN.json             the same blocks, labeled
├── 3_structured/<tag>/sampleN.json          nested into the DMP Tool schema
├── 4_final/<tag>/sampleN.json               annotation rules applied
├── 5_rda/sampleN.json                       RDA JSON, from the same text
└── 6_merged/sampleN.json                    RDA JSON + final narrative JSON
```

`<tag>` is `<model>_<extractor>_whole_doc`. `5_rda` and `6_merged` have no model folder:
a `sampleN.run.json` beside each file names the models that made it, and a run with a
different RDA model replaces the file. Filenames are the same at every stage, so you can
follow one document all the way through.

The RDA prompt experiments (every prompt version, its results and evaluation) are kept
separately in `notebooks/rda-experiments/`.

📚 **More info:** configuration, extraction, scoring, troubleshooting and the pipeline
internals are all documented in **[docs/](docs/README.md)**.

---

## License
This work is licensed under the **[MIT License](https://opensource.org/license/mit/)**. See **[LICENSE](https://github.com/fairdataihub/dmpbridge/blob/main/LICENSE)** for more information.


---

## Feedback and contribution
Use **[GitHub Issues](https://github.com/fairdataihub/dmpbridge/issues)** to submit feedback, report problems, or suggest improvements.  
You can also **fork** the repository and submit a **Pull Request** with your changes.

---

## How to cite
If you use this code, please cite this repository using the **versioned DOI on Zenodo** for the specific release you used (instructions will be added once the Zenodo record is available). For now, you can reference the repository here: **[fairdataihub/dmpchef](https://github.com/fairdataihub/dmpbridge)**.
