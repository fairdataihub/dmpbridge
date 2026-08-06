# Pipeline

```mermaid
flowchart TD
    PDF["<b>DMP PDF</b>"]

    PDF --> READ["<b>Read the PDF</b><br/><small>pdfplumber · Docling · LightOnOCR</small>"]
    READ --> S1["<b>1. Text blocks</b>"]
    S1 --> LABEL["<b>Label each block</b><br/><small>llama3.1:8b · gemma4:e4b · llama3.3:70b</small>"]
    LABEL --> S2["<b>2. Labeled blocks</b>"]
    S2 --> BUILD["<b>Build the structure</b>"]
    BUILD --> S3["<b>3. Structured JSON</b>"]

    S3 --> PATHA["<b>Path A</b><br/>score vs old annotation"]
    S3 --> RULES["<b>Apply the rules</b><br/><small>Rules.xlsx</small>"]
    RULES --> S4["<b>4. Final JSON</b>"]
    S4 --> PATHB["<b>Path B</b><br/>score vs new annotation"]

    classDef input  fill:#1E406E,stroke:#1E406E,color:#ffffff
    classDef step   fill:#F4F7FB,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef cached fill:#ffffff,stroke:#0F766E,stroke-width:2px,color:#111
    classDef data   fill:#ffffff,stroke:#1E406E,stroke-width:2px,color:#111
    classDef rules  fill:#ffffff,stroke:#B45309,stroke-width:2px,color:#111
    classDef pathA  fill:#EDF3FA,stroke:#3C6FA8,stroke-width:2px,color:#111
    classDef pathB  fill:#FDF4E9,stroke:#B45309,stroke-width:2px,color:#111

    class PDF input
    class READ,LABEL,BUILD,RULES step
    class S1 cached
    class S2,S3 data
    class S4 rules
    class PATHA pathA
    class PATHB pathB
```

The numbered boxes are the four things written to disk. The grey boxes between them are the
steps that produce each one.

**Two things worth noticing:**

- **Step 1 is shared.** Reading the PDF doesn't depend on which model does the labeling, so
  it is done once and reused. Labeling with three models costs one read, not three.
- **The two scores branch at step 3.** Path A scores the structure as the model produced it.
  Path B first fills in blank questions using the rules, then scores that. Both are kept, so
  you can compare them.

---

## Detail

| Step | What it does | Choices |
|---|---|---|
| Read the PDF | turn the page into text blocks | `pdfplumber`, `Docling`, `LightOnOCR` |
| Label each block | one LLM call per document | `llama3.1:8b`, `gemma4:e4b`, `llama3.3:70b` |
| Build the structure | nest into sections → questions → answers | — |
| Apply the rules | fill blank questions | `data/input/Rules.xlsx` |

Where each stage is written:

```
data/output/
├── 1_extracted/<extractor>/sampleN.json     keyed by extractor — shared across models
├── 2_labeled/<tag>/sampleN.json
├── 3_structured/<tag>/sampleN.json          <- Path A scores this
└── 4_final/<tag>/sampleN.json               <- Path B scores this
```

`<tag>` is `<model>_<extractor>_whole_doc`. Filenames are the same at every stage, so you
can open `sampleN.json` in each folder and follow one document through.

Both paths use the same scoring — match on shared words, then precision / recall / F1.
Nothing differs but which annotation they compare against.
