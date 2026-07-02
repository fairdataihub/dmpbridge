"""Add Qwen2.5-VL 7B (M09) to 000_experiment_log and 003_strategy_comparison."""
import json
from pathlib import Path

# ── 000_experiment_log.ipynb ─────────────────────────────────────────────────
nb = json.loads(Path("c:/Users/Nahid/dmpbridge/notebooks/000_experiment_log.ipynb")
                .read_bytes().lstrip(b"\xef\xbb\xbf"))
cells = nb["cells"]

def find_cell(keyword):
    for i, c in enumerate(cells):
        if keyword in "".join(c.get("source", [])):
            return i, "".join(c.get("source", []))
    return None, None

# Registry — add M09 row
reg_i, reg_src = find_cell("| M08 |")
if reg_i is not None and "M09" not in reg_src:
    m09_row = (
        "| M09 | Qwen2.5-VL 7B | vision-batch | `qwen2.5vl-7b_vision` "
        "| 10/10 | — | Done |\n"
    )
    new_src = reg_src.replace(
        "| F01 |",
        m09_row + "| F01 |"
    )
    # Update footer note
    new_src = new_src.replace(
        "Page images (M08):",
        "Page images (M08/M09):"
    )
    cells[reg_i]["source"] = new_src
    print("[000] Added M09 row to registry")

# Claude section — add M09 section after M08
claude_i, claude_src = find_cell("### M08 — Vision Batch")
if claude_i is not None and "### M09" not in claude_src:
    m09_section = """

---

## Qwen2.5-VL 7B (local — Ollama)

**Provider:** Ollama (local)
**Model ID:** `qwen2.5vl:7b`
**Evaluation notebook:** `notebooks/003_strategy_comparison.ipynb`

### M09 — Vision Batch

**Output files:** `data/output/labeled/qwen2.5vl-7b_vision/sampleN.json`
**Page images:** `data/output/page_images/sampleN/page_NNN.png` (shared with M08)
**Inference:** `dmpbridge-experiment experiments/qwen2-vl-vision.yaml`
**Note:** Same vision_batch strategy as M08 but running locally via Ollama — no API cost. Images resized to ≤900 px before sending to stay within the 7B model context window.

| Metric | Value |
|---|---|
| Samples complete | 10/10 |
| Block granularity | Line-level (pdfplumber) |
| Bounding boxes | Yes |
| Cost | Free (local) |

Accuracy results pending full evaluation in `003_strategy_comparison.ipynb`."""
    cells[claude_i]["source"] = claude_src.rstrip() + m09_section
    print("[000] Added M09 section (Qwen2.5-VL) to model cells")

# Cross-model summary — add Qwen2.5-VL row
summary_i, summary_src = find_cell("| Llama 3.1 8B |")
if summary_i is not None and "Qwen2.5-VL" not in summary_src:
    qwen_row = "| Qwen2.5-VL 7B | — | — | — | **Vision Batch** | Vision Batch (local) |\n"
    new_src = summary_src.replace(
        "| Llama 3.1 8B |",
        qwen_row + "| Llama 3.1 8B |"
    )
    cells[summary_i]["source"] = new_src
    print("[000] Added Qwen2.5-VL row to cross-model summary")

Path("c:/Users/Nahid/dmpbridge/notebooks/000_experiment_log.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("[000] Saved")

# ── 003_strategy_comparison.ipynb ────────────────────────────────────────────
nb3 = json.loads(Path("c:/Users/Nahid/dmpbridge/notebooks/003_strategy_comparison.ipynb")
                 .read_bytes().lstrip(b"\xef\xbb\xbf"))
cells3 = nb3["cells"]

def find_cell3(keyword):
    for i, c in enumerate(cells3):
        if keyword in "".join(c.get("source", [])):
            return i, "".join(c.get("source", []))
    return None, None

# c01 — add qwen2.5vl to MODEL_FAMILIES
c01_i, c01_src = find_cell3("MODEL_FAMILIES")
if c01_i is not None and "qwen2.5vl" not in c01_src:
    qwen_entry = (
        '    "qwen2.5vl:7b": {\n'
        '        "vision-batch": {"tag": "qwen2.5vl-7b_vision", "color": "#0e7490", "hatch": "xx"},\n'
        '    },\n'
    )
    new_src = c01_src.replace(
        '"claude-opus-4-8": {',
        qwen_entry + '    "claude-opus-4-8": {'
    )
    cells3[c01_i]["source"] = new_src
    print("[003] Added qwen2.5vl:7b to MODEL_FAMILIES")

# m01 header — add Qwen2.5-VL to models table
m01_i, m01_src = find_cell3("| claude-opus-4-8 |")
if m01_i is not None and "qwen2.5vl" not in m01_src:
    qwen_model_row = (
        "| qwen2.5vl:7b | Ollama | — | — | — | `qwen2.5vl-7b_vision` |\n"
    )
    new_src = m01_src.replace(
        "| claude-opus-4-8 |",
        qwen_model_row + "| claude-opus-4-8 |"
    )
    cells3[m01_i]["source"] = new_src
    print("[003] Added qwen2.5vl:7b to models table in header")

Path("c:/Users/Nahid/dmpbridge/notebooks/003_strategy_comparison.ipynb").write_text(
    json.dumps(nb3, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("[003] Saved")
print("\nDone.")
