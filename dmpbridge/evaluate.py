"""Evaluate LLM-labeled JSON against manually labeled ground truth.

Usage (CLI):
    dmpbridge-evaluate                                          # all samples
    dmpbridge-evaluate data/llmlabeled/sample1_llama3.3-70b_batch.json

Notebook usage:
    from dmpbridge.evaluate import (
        load_method, compute_f1_rows,
        extract_gold, evaluate_sample, match,
        LABELS, SHORT, LLM_DIR, MANUAL_DIR, NO_MATCH,
    )
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from . import config as _config
from .logging_setup import get_logger, setup_logging
from .prompt import LABELS

logger = get_logger(__name__)

SHORT = ["title", "sec.title", "sec.desc", "q.text", "ans.text"]

_ROOT      = Path(__file__).parent.parent
MANUAL_DIR = _ROOT / "data/manuallabeled"
LLM_DIR    = _ROOT / "data/llmlabeled"

MODEL_TAG  = _config.MODEL.replace(":", "-").replace("/", "-")
LLM_SUFFIX = f"_{MODEL_TAG}_batch"


# ── Text helpers ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    """Lowercase and split text into a set of alphanumeric word tokens."""
    return set(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def containment(block_tokens: set, gold_tokens: set) -> float:
    """Fraction of block tokens that appear in the gold text."""
    if not block_tokens:
        return 0.0
    return len(block_tokens & gold_tokens) / len(block_tokens)


# ── Load manual labels ────────────────────────────────────────────────────────

def extract_gold(path: Path) -> list[tuple[str, str]]:
    """Read a manually labeled DMP JSON and return a flat list of (text, label) pairs."""
    data     = json.loads(path.read_text(encoding="utf-8"))
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
            ans_text = ""
            if isinstance(ans, dict):
                ans_text = ans.get("json", {}).get("answer", "") or ans.get("text", "")
            if ans_text:
                pairs.append((ans_text.strip(), "answer.text"))

    return pairs


# ── Match a block to the best gold label ──────────────────────────────────────

NO_MATCH = "__no_match__"


def match(block_text: str, gold_pairs: list[tuple[str, str]]) -> str | None:
    """Return the gold label for this block, or None if containment < 0.75."""
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
    """Build a confusion matrix comparing predicted blocks against gold pairs.

    Forward check: for each predicted block, find its best-matching gold label.
    Reverse check: for each gold item, record it as missed if no predicted block covers it.
    Returns {true_label: {pred_label: count}}.
    """
    blocks    = json.loads(pred_path.read_text(encoding="utf-8"))
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for block in blocks:
        gold_label = match(block["text"], gold_pairs) or NO_MATCH
        confusion[gold_label][block.get("label", "answer.text")] += 1

    pred_texts = [b["text"] for b in blocks]
    for gold_text, gold_label in gold_pairs:
        gtok    = tokenize(gold_text)
        matched = any(
            containment(tokenize(pt), gtok) >= 0.75
            for pt in pred_texts
            if tokenize(pt)
        )
        if not matched:
            confusion[gold_label]["__missed__"] += 1

    return confusion


# ── Gold-based metrics ───────────────────────────────────────────────────────

def gold_metrics(confusion: dict) -> tuple[int, int, int]:
    """Compute gold-based metrics using the same manual-label denominator for all strategies.

    Unlike block accuracy (correct / predicted_blocks), gold accuracy uses the
    total number of manually labeled gold items as the denominator, making it
    directly comparable across strategies that produce different block counts
    (e.g. pdfplumber line-level vs Claude PDF-direct paragraph-level).

    Returns
    -------
    (correct, covered, total_gold)
        correct    : gold items covered AND correctly labeled
        covered    : gold items matched by at least one predicted block
        total_gold : all gold items (correct + mislabeled + missed)
    """
    correct    = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
    missed     = sum(confusion.get(lbl, {}).get("__missed__", 0) for lbl in LABELS)
    total_pred = sum(
        sum(v for k, v in confusion.get(lbl, {}).items() if k != "__missed__")
        for lbl in LABELS
    )
    total_gold = total_pred + missed
    covered    = total_gold - missed
    return correct, covered, total_gold


# ── Aggregate confusion matrices ──────────────────────────────────────────────

def add_confusion(total: dict, new: dict) -> None:
    """Accumulate new confusion matrix counts into total (mutates total in place)."""
    for true_lbl, preds in new.items():
        for pred_lbl, count in preds.items():
            total[true_lbl][pred_lbl] += count


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_label_metrics(confusion: dict) -> list[dict]:
    """Compute precision, recall, and F1 for each label. Returns a list of dicts."""
    rows = []
    for lbl in LABELS:
        tp      = confusion.get(lbl, {}).get(lbl, 0)
        fp      = sum(confusion.get(other, {}).get(lbl, 0) for other in LABELS if other != lbl)
        fn      = sum(v for pred, v in confusion.get(lbl, {}).items() if pred != lbl)
        support = tp + fn
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / support   if support   else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"label": lbl, "precision": p, "recall": r, "f1": f1, "support": support})
    return rows


def compute_f1_rows(confusion: dict):
    """Return a pandas DataFrame with precision, recall, F1, and support per label."""
    import pandas as pd
    return pd.DataFrame(_compute_label_metrics(confusion))


# ── Notebook helper ───────────────────────────────────────────────────────────

def _snum(path: Path) -> int:
    """Extract sample number from a path like data/manuallabeled/sample3_dmp.json → 3."""
    return int(path.stem.replace("_dmp", "").replace("sample", ""))


def load_method(tag: str):
    """Load and evaluate all samples for a given file tag.

    Files follow the pattern: ``data/llmlabeled/sampleN_{tag}.json``

    Returns
    -------
    tuple[pd.DataFrame, dict, pd.DataFrame] | tuple[None, None, None]
        ``(accuracy_df, confusion_dict, errors_df)`` or ``(None, None, None)``
        if no output files are found.
    """
    import pandas as pd

    samples = sorted(MANUAL_DIR.glob("*_dmp.json"), key=_snum)
    found   = [mp for mp in samples
               if (LLM_DIR / f"{mp.stem.replace('_dmp', '')}_{tag}.json").exists()]
    if not found:
        return None, None, None

    conf_all: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows, errors = [], []

    for mp in samples:
        stem = mp.stem.replace("_dmp", "")
        pp   = LLM_DIR / f"{stem}_{tag}.json"
        if not pp.exists():
            continue
        gold   = extract_gold(mp)
        conf   = evaluate_sample(pp, gold)
        blocks = json.loads(pp.read_text(encoding="utf-8"))
        for b in blocks:
            tl = match(b["text"], gold)
            pl = b.get("label", "answer.text")
            conf_all[tl or NO_MATCH][pl] += 1
            if tl != pl:
                errors.append({
                    "sample": stem,
                    "text":   b["text"][:120],
                    "true":   tl or "no_match",
                    "pred":   pl,
                    "page":   b.get("page", "-"),
                })
        tp = sum(conf.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
        n  = sum(sum(v.values()) for v in conf.values())
        rows.append({
            "sample":   stem,
            "total":    n,
            "correct":  tp,
            "errors":   n - tp,
            "accuracy": tp / n if n else 0,
            "formula":  f"{tp}/{n}",
        })

    return pd.DataFrame(rows), conf_all, pd.DataFrame(errors)


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_confusion(confusion: dict) -> None:
    """Print confusion matrix. Correct predictions marked *, missed items marked !."""
    col_w  = 10
    label  = "true \\ pred"
    header = f"{label:<18}" + "".join(f"{s:>{col_w}}" for s in SHORT) + f"{'missed':>{col_w}}"
    print(header)
    print("-" * len(header))
    for true_lbl, short_true in zip(LABELS, SHORT):
        row    = confusion.get(true_lbl, {})
        total  = sum(row.values())
        cells  = ""
        for pred_lbl in LABELS:
            n      = row.get(pred_lbl, 0)
            marker = f"{n}*" if pred_lbl == true_lbl and n else str(n)
            cells += f"{marker:>{col_w}}"
        missed     = row.get("__missed__", 0)
        missed_str = f"{missed}!" if missed else "0"
        print(f"{short_true:<18}{cells}{missed_str:>{col_w}}  (n={total})")


def print_f1(confusion: dict) -> None:
    """Print precision, recall, and F1 for each label."""
    print(f"\n{'Label':<22}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("-" * 62)
    total_tp = total_support = 0
    for row in _compute_label_metrics(confusion):
        lbl, p, r, f1, support = (
            row["label"], row["precision"], row["recall"], row["f1"], row["support"]
        )
        print(f"{lbl:<22}  {p*100:>9.1f}%  {r*100:>7.1f}%  {f1*100:>7.1f}%  {support:>8}")
        total_tp      += confusion.get(lbl, {}).get(lbl, 0)
        total_support += support
    overall = total_tp / total_support if total_support else 0.0
    print("-" * 62)
    print(f"{'Overall accuracy':<22}  {'':>10}  {'':>8}  {overall*100:>7.1f}%  {total_support:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _sample_header() -> None:
    """Print the per-sample accuracy table header."""
    print(f"{'Sample':<12}  {'Total':>5}  {'Correct':>7}  {'Errors':>6}  {'Accuracy':>8}  Formula")
    print("-" * 58)


def _sample_row(stem: str, n: int, tp: int) -> None:
    """Print one row of the per-sample accuracy table."""
    errors = n - tp
    acc    = tp / n * 100 if n else 0.0
    print(f"{stem:<12}  {n:>5}  {tp:>7}  {errors:>6}  {acc:>7.1f}%  {tp}/{n}")


def print_missed(pred_path: Path, gold_pairs: list[tuple[str, str]]) -> None:
    """Print every gold item for which the LLM produced no matching block."""
    blocks     = json.loads(pred_path.read_text(encoding="utf-8"))
    pred_texts = [b["text"] for b in blocks]
    missed     = []
    for gold_text, gold_label in gold_pairs:
        gtok    = tokenize(gold_text)
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


def run_single(pred_path: Path) -> None:
    """Evaluate one LLM output file against its matching manual annotation."""
    stem        = pred_path.stem.split("_")[0]
    manual_path = MANUAL_DIR / f"{stem}_dmp.json"
    if not manual_path.exists():
        logger.warning("No manual label found for %s at %s", stem, manual_path)
        return
    gold      = extract_gold(manual_path)
    confusion = evaluate_sample(pred_path, gold)
    tp = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
    n  = sum(sum(v.values()) for v in confusion.values())
    print("\n" + "=" * 62)
    print(f"  {pred_path.name}  vs  {manual_path.name}")
    print("=" * 62 + "\n")
    _sample_header()
    _sample_row(stem, n, tp)
    print("\nConfusion matrix  (* = correct, ! = missed)\n")
    print_confusion(confusion)
    print_f1(confusion)
    print("\nMissed gold items (LLM produced no matching block):")
    print_missed(pred_path, gold)


def run_all() -> None:
    """Evaluate all samples: per-sample accuracy, missed items, aggregate confusion matrix."""
    total_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples = sorted(MANUAL_DIR.glob("*_dmp.json"))

    _sample_header()
    total_n = total_tp = 0
    per_sample = []

    for manual_path in samples:
        stem      = manual_path.stem.replace("_dmp", "")
        pred_path = LLM_DIR / f"{stem}{LLM_SUFFIX}.json"
        if not pred_path.exists():
            logger.warning("SKIP %s — no %s", stem, pred_path.name)
            continue
        gold      = extract_gold(manual_path)
        confusion = evaluate_sample(pred_path, gold)
        add_confusion(total_confusion, confusion)

        tp         = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
        n          = sum(sum(v.values()) for v in confusion.values())
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

    print("\n" + "=" * 62)
    print("  MISSED ITEMS PER SAMPLE")
    print("=" * 62)
    for stem, pred_path, gold, mismatch_n, missed_n in per_sample:
        if missed_n == 0 and mismatch_n == 0:
            continue
        print(f"\n{stem}  ({mismatch_n} mismatch, {missed_n} missed)")
        print_missed(pred_path, gold)

    print("\n" + "=" * 62)
    print("  AGGREGATE  (all samples pooled)")
    print("=" * 62)
    print("\nConfusion matrix  (* = correct, ! = missed)\n")
    print_confusion(total_confusion)
    print_f1(total_confusion)


def main() -> None:
    """CLI entry point: dmpbridge-evaluate."""
    setup_logging()
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if pos_args:
        run_single(Path(pos_args[0]))
    else:
        run_all()


if __name__ == "__main__":
    main()
