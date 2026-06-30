"""
Evaluate LLM-labeled JSON against manually labeled ground truth.

Usage:
    python evaluate.py                                      # evaluate all samples
    python evaluate.py data/llmlabeled/sample1_llama3.3-70b.json  # single file
"""

#
#   manual JSON (gold)              LLM JSON (predicted)
#         │                               │
#         ▼                               ▼
#   extract_gold()              load flat block list
#   returns (text, label) pairs
#         │                               │
#         └───────────────┬───────────────┘
#                         ▼
#                 evaluate_sample()
#                 │
#                 ├── forward check:  each predicted block → best matching gold label
#                 └── reverse check:  each gold item → any block covers it? (missed?)
#                         │
#                         ▼
#                 confusion matrix  { true_label: { pred_label: count } }
#                         │
#                         ▼
#         print accuracy / F1 per label / missed items
#

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dmpbridge import config as _config

# The 5 label names and their short display versions used in the confusion matrix table.
LABELS = ["title", "section.title", "section.description", "question.text", "answer.text"]
SHORT  = ["title", "sec.title", "sec.desc", "q.text", "ans.text"]

_ROOT      = Path(__file__).parent
MANUAL_DIR = _ROOT / "data/manuallabeled"
LLM_DIR    = _ROOT / "data/llmlabeled"

MODEL_TAG  = _config.MODEL.replace(":", "-").replace("/", "-")
LLM_SUFFIX = f"_{MODEL_TAG}"


# ── Text helpers ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    # Convert text to a set of lowercase words, stripping all punctuation.
    return set(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def containment(block_tokens: set, gold_tokens: set) -> float:
    """Measure how much of a predicted block's vocabulary is covered by a gold item. Returns 1.0 if all block tokens appear in the gold text, 0.0 if none do."""
    if not block_tokens:
        return 0.0
    return len(block_tokens & gold_tokens) / len(block_tokens)


# ── Load manual labels ────────────────────────────────────────────────────────

def extract_gold(path: Path) -> list[tuple[str, str]]:
    """Read a manually labeled DMP JSON and return a flat list of (text, label) pairs. Walks the nested schema: title → section titles → section descriptions → question texts → answer texts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    template = data.get("narrative", data).get("template", {})
    pairs: list[tuple[str, str]] = []

    # Collect the document title.
    title = template.get("title", "").strip()
    if title:
        pairs.append((title, "title"))

    # Walk every section → question → answer to collect all labeled text.
    for section in template.get("section", []):
        if section.get("title"):
            pairs.append((section["title"].strip(), "section.title"))
        if section.get("description"):
            pairs.append((section["description"].strip(), "section.description"))
        for question in section.get("question", []):
            if question.get("text"):
                pairs.append((question["text"].strip(), "question.text"))
            ans = question.get("answer", {})
            # Answer text is nested under answer → json → answer.
            ans_text = ""
            if isinstance(ans, dict):
                ans_text = ans.get("json", {}).get("answer", "") or ans.get("text", "")
            if ans_text:
                pairs.append((ans_text.strip(), "answer.text"))

    return pairs


# ── Match a block to the best gold label ──────────────────────────────────────

NO_MATCH = "__no_match__"


def match(block_text: str, gold_pairs: list[tuple[str, str]]) -> str | None:
    """Find the gold item that best contains this predicted block's tokens. Returns the gold label if containment >= 0.75, otherwise None (no match)."""
    btok = tokenize(block_text)
    if not btok:
        return None
    best_score, best_label = 0.0, None
    for gold_text, gold_label in gold_pairs:
        score = containment(btok, tokenize(gold_text))
        if score > best_score:
            best_score, best_label = score, gold_label
    return best_label if best_score >= 0.75 else None


# ── Evaluate one sample ───────────────────────────────────────────────────────

def evaluate_sample(pred_path: Path, gold_pairs: list[tuple[str, str]]) -> dict:
    """Compare all predicted blocks against gold pairs and build a confusion matrix. 1. For each predicted block, find the best matching gold item (forward check). 2. For each gold item, check if any predicted block covers it (reverse check for missed items). Returns a dict of {true_label: {pred_label: count}}."""
    blocks = json.loads(pred_path.read_text(encoding="utf-8"))
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Forward check — for each predicted block, find its gold label and record the prediction.
    for block in blocks:
        gold_label = match(block["text"], gold_pairs) or NO_MATCH
        pred_label = block.get("label", "answer.text")
        confusion[gold_label][pred_label] += 1

    # Reverse check — for each gold item, check if any predicted block covered it.
    # If not, it means the LLM missed that piece of content entirely.
    pred_texts = [b["text"] for b in blocks]
    for gold_text, gold_label in gold_pairs:
        gtok = tokenize(gold_text)
        matched = any(
            containment(tokenize(pt), gtok) >= 0.75
            for pt in pred_texts
            if tokenize(pt)
        )
        if not matched:
            confusion[gold_label]["__missed__"] += 1

    return confusion


# ── Aggregate confusion matrices ──────────────────────────────────────────────

def add_confusion(total, new):
    # Merge a per-sample confusion matrix into the running total across all samples.
    for true_lbl, preds in new.items():
        for pred_lbl, count in preds.items():
            total[true_lbl][pred_lbl] += count


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_confusion(confusion: dict):
    # Print a grid showing true labels as rows and predicted labels as columns.
    # Correct predictions are marked with *, missed items with !.
    col_w = 10
    label = "true \\ pred"
    header = f"{label:<18}" + "".join(f"{s:>{col_w}}" for s in SHORT) + f"{'missed':>{col_w}}"
    print(header)
    print("-" * len(header))
    for true_lbl, short_true in zip(LABELS, SHORT):
        row = confusion.get(true_lbl, {})
        total = sum(row.values())
        cells = ""
        for pred_lbl in LABELS:
            n = row.get(pred_lbl, 0)
            marker = f"{n}*" if pred_lbl == true_lbl and n else str(n)
            cells += f"{marker:>{col_w}}"
        missed = row.get("__missed__", 0)
        missed_str = f"{missed}!" if missed else "0"
        print(f"{short_true:<18}{cells}{missed_str:>{col_w}}  (n={total})")


def print_f1(confusion: dict):
    # Compute and print precision, recall, and F1 for each label.
    print(f"\n{'Label':<22}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("-" * 62)
    total_tp = total_support = 0
    for lbl in LABELS:
        tp = confusion.get(lbl, {}).get(lbl, 0)
        fp = sum(confusion.get(other, {}).get(lbl, 0) for other in LABELS if other != lbl)
        fn = sum(v for pred, v in confusion.get(lbl, {}).items() if pred != lbl)
        support = tp + fn
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / support   if support   else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"{lbl:<22}  {p*100:>9.1f}%  {r*100:>7.1f}%  {f1*100:>7.1f}%  {support:>8}")
        total_tp += tp
        total_support += support
    overall = total_tp / total_support if total_support else 0.0
    print("-" * 62)
    print(f"{'Overall accuracy':<22}  {'':>10}  {'':>8}  {overall*100:>7.1f}%  {total_support:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _sample_header():
    print(f"{'Sample':<12}  {'Total':>5}  {'Correct':>7}  {'Errors':>6}  {'Accuracy':>8}  Formula")
    print("-" * 58)


def _sample_row(stem: str, n: int, tp: int) -> None:
    errors = n - tp
    acc    = tp / n * 100 if n else 0.0
    print(f"{stem:<12}  {n:>5}  {tp:>7}  {errors:>6}  {acc:>7.1f}%  {tp}/{n}")


def print_missed(pred_path: Path, gold_pairs: list[tuple[str, str]]) -> None:
    """Print every gold item that the LLM produced no matching block for."""
    blocks = json.loads(pred_path.read_text(encoding="utf-8"))
    pred_texts = [b["text"] for b in blocks]
    missed = []
    for gold_text, gold_label in gold_pairs:
        gtok = tokenize(gold_text)
        matched = any(
            containment(tokenize(pt), gtok) >= 0.75
            for pt in pred_texts
            if tokenize(pt)
        )
        if not matched:
            missed.append((gold_label, gold_text))
    if not missed:
        print("  (none)")
        return
    for lbl, text in missed:
        print(f"  [{lbl}]  {repr(text[:80])}")


def run_single(pred_path: Path):
    """Evaluate one LLM output file against its matching manual annotation and print full results."""
    stem = pred_path.stem.split("_")[0]  # e.g. "sample1"
    manual_path = MANUAL_DIR / f"{stem}_dmp.json"
    if not manual_path.exists():
        print(f"No manual label found for {stem} at {manual_path}")
        return
    gold = extract_gold(manual_path)
    confusion = evaluate_sample(pred_path, gold)

    tp = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
    n  = sum(sum(v.values()) for v in confusion.values())
    print(f"\n{'='*62}")
    print(f"  {pred_path.name}  vs  {manual_path.name}")
    print(f"{'='*62}\n")
    _sample_header()
    _sample_row(stem, n, tp)
    print()
    print("\nConfusion matrix  (* = correct, ! = missed)\n")
    print_confusion(confusion)
    print_f1(confusion)
    print(f"\nMissed gold items (LLM produced no matching block):")
    print_missed(pred_path, gold)


def run_all():
    """Evaluate all samples, print per-sample accuracy, missed items, and an aggregate confusion matrix."""
    total_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples = sorted(MANUAL_DIR.glob("*_dmp.json"))

    _sample_header()
    total_n = total_tp = 0
    per_sample = []

    for manual_path in samples:
        stem = manual_path.stem.replace("_dmp", "")
        pred_path = LLM_DIR / f"{stem}{LLM_SUFFIX}.json"
        if not pred_path.exists():
            print(f"SKIP {stem} — no {pred_path.name}")
            continue
        gold = extract_gold(manual_path)
        confusion = evaluate_sample(pred_path, gold)
        add_confusion(total_confusion, confusion)

        tp = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
        n  = sum(sum(v.values()) for v in confusion.values())
        missed_n   = sum(confusion.get(lbl, {}).get("__missed__", 0) for lbl in LABELS)
        mismatch_n = n - tp - missed_n

        if n:
            _sample_row(stem, n, tp)
        else:
            print(f"{stem:<12}  no blocks")
        total_n  += n
        total_tp += tp
        per_sample.append((stem, pred_path, gold, mismatch_n, missed_n))

    print("-" * 58)
    _sample_row("TOTAL", total_n, total_tp)

    # Print which specific gold items were missed in each sample.
    print(f"\n{'='*62}")
    print("  MISSED ITEMS PER SAMPLE")
    print(f"{'='*62}")
    for stem, pred_path, gold, mismatch_n, missed_n in per_sample:
        if missed_n == 0 and mismatch_n == 0:
            continue
        print(f"\n{stem}  ({mismatch_n} mismatch, {missed_n} missed)")
        print_missed(pred_path, gold)

    # Print the confusion matrix and F1 scores pooled across all samples.
    print(f"\n{'='*62}")
    print("  AGGREGATE  (all samples pooled)")
    print(f"{'='*62}")
    print("\nConfusion matrix  (* = correct, ! = missed)\n")
    print_confusion(total_confusion)
    print_f1(total_confusion)


if __name__ == "__main__":
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if pos_args:
        run_single(Path(pos_args[0]))
    else:
        run_all()
