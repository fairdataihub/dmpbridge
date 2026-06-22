"""
Evaluate LLM-labeled JSON against manually labeled ground truth.

Usage:
    python evaluate.py                       # evaluate all samples, print confusion matrix + F1
    python evaluate.py data/llmlabeled/sample1_improved.json   # single file
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LABELS = ["title", "section.title", "section.description", "question.text", "answer.text"]
SHORT  = ["title", "sec.title", "sec.desc", "q.text", "ans.text"]

_ROOT      = Path(__file__).parent
MANUAL_DIR = _ROOT / "data/manuallabeled"
LLM_DIR    = _ROOT / "data/llmlabeled"


# ── Text helpers ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def containment(block_tokens: set, gold_tokens: set) -> float:
    """Fraction of block tokens that appear in the gold text."""
    if not block_tokens:
        return 0.0
    return len(block_tokens & gold_tokens) / len(block_tokens)


# ── Load manual labels ────────────────────────────────────────────────────────

def extract_gold(path: Path) -> list[tuple[str, str]]:
    """Return list of (text, label) from a manual-labeled DMP JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    template = data.get("narrative", data).get("template", {})
    pairs: list[tuple[str, str]] = []

    title = template.get("title", "").strip()
    if title:
        pairs.append((title, "title"))

    for section in template.get("section", []):
        if section.get("title"):
            pairs.append((section["title"].strip(), "section.title"))
        if section.get("description"):
            pairs.append((section["description"].strip(), "section.description"))
        for question in section.get("question", []):
            if question.get("text"):
                pairs.append((question["text"].strip(), "question.text"))
            ans = question.get("answer", {})
            # answer text lives at answer["json"]["answer"]
            ans_text = ""
            if isinstance(ans, dict):
                ans_text = ans.get("json", {}).get("answer", "") or ans.get("text", "")
            if ans_text:
                pairs.append((ans_text.strip(), "answer.text"))

    return pairs


# ── Match a block to the best gold label ─────────────────────────────────────

def match(block_text: str, gold_pairs: list[tuple[str, str]]) -> str | None:
    """
    Return the gold label whose text best contains the block.
    Uses containment (block tokens ⊆ gold tokens) so that short blocks
    extracted from long answer/description paragraphs match correctly.
    Requires ≥ 75% containment and at least 4 block tokens.
    """
    btok = tokenize(block_text)
    if len(btok) < 4:
        return None

    best_score, best_label = 0.0, None
    for gold_text, gold_label in gold_pairs:
        gtok = tokenize(gold_text)
        score = containment(btok, gtok)
        if score > best_score:
            best_score, best_label = score, gold_label

    return best_label if best_score >= 0.75 else None


# ── Evaluate one sample ───────────────────────────────────────────────────────

def evaluate_sample(pred_path: Path, gold_pairs: list[tuple[str, str]]) -> dict:
    """Return confusion dict {true_label: {pred_label: count}}."""
    blocks = json.loads(pred_path.read_text(encoding="utf-8"))
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for block in blocks:
        gold_label = match(block["text"], gold_pairs)
        if gold_label is None:
            continue
        pred_label = block.get("label", "answer.text")
        confusion[gold_label][pred_label] += 1

    return confusion


# ── Aggregate confusion matrices ──────────────────────────────────────────────

def add_confusion(total, new):
    for true_lbl, preds in new.items():
        for pred_lbl, count in preds.items():
            total[true_lbl][pred_lbl] += count


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_confusion(confusion: dict):
    col_w = 10
    label = "true \\ pred"
    header = f"{label:<18}" + "".join(f"{s:>{col_w}}" for s in SHORT)
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
        print(f"{short_true:<18}{cells}  (n={total})")


def print_f1(confusion: dict):
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

def run_single(pred_path: Path):
    stem = pred_path.stem.split("_")[0]  # e.g. "sample1"
    manual_path = MANUAL_DIR / f"{stem}_dmp.json"
    if not manual_path.exists():
        print(f"No manual label found for {stem} at {manual_path}")
        return
    gold = extract_gold(manual_path)
    confusion = evaluate_sample(pred_path, gold)

    print(f"\n{'='*62}")
    print(f"  {pred_path.name}  vs  {manual_path.name}")
    print(f"{'='*62}")
    print("\nConfusion matrix  (* = correct)\n")
    print_confusion(confusion)
    print_f1(confusion)


def run_all():
    total_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples = sorted(MANUAL_DIR.glob("*_dmp.json"))

    for manual_path in samples:
        stem = manual_path.stem.replace("_dmp", "")
        pred_path = LLM_DIR / f"{stem}_improved.json"
        if not pred_path.exists():
            print(f"SKIP {stem} — no {pred_path.name}")
            continue
        gold = extract_gold(manual_path)
        confusion = evaluate_sample(pred_path, gold)
        add_confusion(total_confusion, confusion)

        # Per-sample accuracy
        tp = sum(confusion.get(l, {}).get(l, 0) for l in LABELS)
        n  = sum(sum(v.values()) for v in confusion.values())
        print(f"{stem:<12}  matched={n:>3}  accuracy={tp/n*100:>5.1f}%" if n else f"{stem:<12}  no matches")

    print(f"\n{'='*62}")
    print("  AGGREGATE  (all samples pooled)")
    print(f"{'='*62}")
    print("\nConfusion matrix  (* = correct)\n")
    print_confusion(total_confusion)
    print_f1(total_confusion)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(Path(sys.argv[1]))
    else:
        run_all()
