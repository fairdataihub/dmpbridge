# Pipeline

```mermaid
flowchart TD
    PDF["<b>DMP PDF</b><br/>10 documents, 23 pages"]

    PDF --> PLUMB["<b>pdfplumber</b><br/><small>+ line merge</small>"]
    PDF --> DOC["<b>Docling</b><br/><small>ML layout</small>"]
    PDF --> OCR["<b>LightOnOCR</b><br/><small>vision OCR</small>"]

    PLUMB --> S1
    DOC --> S1
    OCR --> S1

    S1["<b>STAGE 1 — extracted blocks</b><br/>
        <code>1_extracted/&lt;extractor&gt;/sampleN.json</code><br/>
        <i>cached per extractor — computed once, reused by every model</i>"]

    S1 -->|"LLM · whole-doc<br/>one call per document"| S2

    S2["<b>STAGE 2 — labeled blocks</b><br/>
        <code>2_labeled/&lt;tag&gt;/sampleN.json</code><br/>
        <i>each block gets one of 5 labels + confidence</i>"]

    S2 -->|"converter.to_structured()"| S3

    S3["<b>STAGE 3 — structured JSON</b><br/>
        <code>3_structured/&lt;tag&gt;/sampleN.json</code><br/>
        <i>sections → questions → answers</i>"]

    S3 --> PATHA["<b>PATH A</b><br/>scored against<br/><code>ground_truth_old_version</code>"]
    S3 -->|"Rules.xlsx<br/>fill empty question.text"| S4

    S4["<b>STAGE 4 — final JSON</b><br/>
        <code>4_final/&lt;tag&gt;/sampleN.json</code>"]

    S4 --> PATHB["<b>PATH B</b><br/>scored against<br/><code>ground_truth_new_version</code>"]

    classDef stage  fill:#ffffff,stroke:#1E406E,stroke-width:2px,color:#111
    classDef cached fill:#ffffff,stroke:#0F766E,stroke-width:2px,color:#111
    classDef rules  fill:#ffffff,stroke:#B45309,stroke-width:2px,color:#111
    classDef extr   fill:#ffffff,stroke:#3C6FA8,stroke-width:1.5px,color:#111
    classDef input  fill:#1E406E,stroke:#1E406E,color:#ffffff
    classDef pathA  fill:#EDF3FA,stroke:#3C6FA8,stroke-width:2px,color:#111
    classDef pathB  fill:#FDF4E9,stroke:#B45309,stroke-width:2px,color:#111

    class PDF input
    class PLUMB,DOC,OCR extr
    class S1 cached
    class S2,S3 stage
    class S4 rules
    class PATHA pathA
    class PATHB pathB
```

**Two things the diagram is meant to make obvious:**

- **Stage 1 is cached and keyed by extractor**, not by model. Extraction doesn't depend on
  which LLM labels the blocks, so labeling with three models costs one extraction rather
  than three.
- **The paths diverge at stage 3.** Path A scores the structured output against the original
  annotation; Path B scores stage 4, after the `Rules.xlsx` conversion, against the newer
  one. Both stages are always written, so the two forms can be compared directly.

Both paths use the same scoring engine — greedy match by token containment ≥ 0.75, then
micro-averaged precision / recall / F1. Nothing differs but the reference data.
