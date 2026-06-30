"""
Evaluate LLM-labeled JSON against manually labeled ground truth.

Usage:
    python evaluate.py                                      # evaluate all samples
    python evaluate.py data/llmlabeled/sample1_llama3.3-70b.json  # single file
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dmpbridge import config as _config

LABELS = ["title", "section.title", "section.description", "question.text", "answer.text"]
SHORT  = ["title", "sec.title", "sec.desc", "q.text", "ans.text"]

_ROOT      = Path(__file__).parent
MANUAL_DIR = _ROOT / "data/manuallabeled"
LLM_DIR    = _ROOT / "data/llmlabeled"

MODEL_TAG  = _config.MODEL.replace(":", "-").replace("/", "-")
LLM_SUFFIX = f"_{MODEL_TAG}"


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

NO_MATCH = "__no_match__"


def _best_gold_idx(btok: set, gold_pairs: list[tuple[str, str]]) -> tuple[int | None, float, str | None]:
    """Return (index, score, label) of best gold match for a block token set."""
    best_score, best_idx, best_label = 0.0, None, None
    for j, (gt, gl) in enumerate(gold_pairs):
        score = containment(btok, tokenize(gt))
        if score > best_score:
            best_score, best_idx, best_label = score, j, gl
    return best_idx, best_score, best_label


# ── Evaluate one sample ───────────────────────────────────────────────────────

def evaluate_sample(pred_path: Path, gold_pairs: list[tuple[str, str]]) -> dict:
    """Return confusion dict {true_label: {pred_label: count}}.

    Forward: each pred block → best gold match (containment ≥ 0.75).
    Reverse: any gold item that was never the best match for any pred block
             is counted as missed. This uses the same forward matching so that
             short gold entries ('Data Repositories.') are not falsely covered
             by tiny pred blocks ('data.') that forward-match something else.
    """
    blocks = json.loads(pred_path.read_text(encoding="utf-8"))
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    covered: set[int] = set()  # gold indices forward-matched by at least one pred block

    for block in blocks:
        btok = tokenize(block["text"])
        if not btok:
            continue
        idx, score, gold_label = _best_gold_idx(btok, gold_pairs)
        if score >= 0.75 and idx is not None:
            covered.add(idx)
            gold_label = gold_label
        else:
            gold_label = NO_MATCH
        pred_label = block.get("label", "answer.text")
        confusion[gold_label][pred_label] += 1

    # Reverse: gold items never forward-matched by any pred block → missed
    for j, (gold_text, gold_label) in enumerate(gold_pairs):
        if j not in covered:
            confusion[gold_label]["__missed__"] += 1

    return confusion


# ── Aggregate confusion matrices ──────────────────────────────────────────────

def add_confusion(total, new):
    for true_lbl, preds in new.items():
        for pred_lbl, count in preds.items():
            total[true_lbl][pred_lbl] += count


# ── Error list helper (used by notebooks) ────────────────────────────────────

def get_errors(pred_path: Path, gold_pairs: list[tuple[str, str]]) -> list[dict]:
    """Return forward mismatches + missed gold items as a list of dicts."""
    blocks = json.loads(pred_path.read_text(encoding="utf-8"))
    errors: list[dict] = []
    covered: set[int] = set()

    for block in blocks:
        btok = tokenize(block["text"])
        if not btok:
            continue
        idx, score, gold_label = _best_gold_idx(btok, gold_pairs)
        if score >= 0.75 and idx is not None:
            covered.add(idx)
            true_label = gold_label
        else:
            true_label = NO_MATCH
        pred_label = block.get("label", "answer.text")
        if true_label != pred_label:
            errors.append({
                "text": block["text"][:120],
                "true": true_label,
                "pred": pred_label,
                "page": block.get("page", "-"),
            })

    for j, (gold_text, gold_label) in enumerate(gold_pairs):
        if j not in covered:
            errors.append({
                "text": gold_text[:120],
                "true": gold_label,
                "pred": "__missed__",
                "page": "-",
            })

    return errors


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_confusion(confusion: dict):
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
    """Print each gold item that no LLM block matched."""
    blocks = json.loads(pred_path.read_text(encoding="utf-8"))
    covered: set[int] = set()
    for block in blocks:
        btok = tokenize(block["text"])
        if not btok:
            continue
        idx, score, _ = _best_gold_idx(btok, gold_pairs)
        if score >= 0.75 and idx is not None:
            covered.add(idx)
    missed = [
        (gl, gt) for j, (gt, gl) in enumerate(gold_pairs) if j not in covered
    ]

    if not missed:
        print("  (none)")
        return
    for lbl, text in missed:
        print(f"  [{lbl}]  {repr(text[:80])}")


def run_single(pred_path: Path):
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
        missed_n = sum(confusion.get(lbl, {}).get("__missed__", 0) for lbl in LABELS)
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

    # ── Per-sample missed detail ──────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  MISSED ITEMS PER SAMPLE")
    print(f"{'='*62}")
    for stem, pred_path, gold, mismatch_n, missed_n in per_sample:
        if missed_n == 0 and mismatch_n == 0:
            continue
        print(f"\n{stem}  ({mismatch_n} mismatch, {missed_n} missed)")
        print_missed(pred_path, gold)

    # ── Aggregate ─────────────────────────────────────────────────────────────
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
