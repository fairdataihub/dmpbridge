"""Evaluate LLM-labeled JSON against manually labeled ground truth.

Usage (CLI):
    dmpbridge-evaluate                                          # all samples
    dmpbridge-evaluate data/llmlabeled/sample1_llama3.3-70b_whole_doc.json

Notebook usage:
    from dmpbridge.evaluation.evaluate import (
        load_method, compute_f1_rows, confusion_matrix_df,
        load_confidence, confidence_calibration_df,
        extract_gold, evaluate_sample, match,
        micro_prf1, print_micro_prf1,
        LABELS, SHORT, LLM_DIR, MANUAL_DIR, NO_MATCH,
    )
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from ..core import config as _config
from ..prompts import LABELS
from ..utils import get_logger, setup_logging

logger = get_logger(__name__)

SHORT = ["title", "sec.title", "sec.desc", "q.text", "ans.text"]

_ROOT           = Path(__file__).parent.parent.parent
MANUAL_DIR      = _ROOT / "data/input/ground_truth_old_version"   # files: sampleN_old_dmp.json
NEW_MANUAL_DIR  = _ROOT / "data/input/ground_truth_new_version"   # files: sampleN_new_dmp.json
LLM_DIR         = _ROOT / "data/output/labeled"

MODEL_TAG  = _config.MODEL.replace(":", "-").replace("/", "-")
LLM_SUFFIX = f"{MODEL_TAG}_whole_doc"


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
        sec_title = section.get("title", "").strip()
        if sec_title:
            pairs.append((sec_title, "section.title"))
        if section.get("description"):
            pairs.append((section["description"].strip(), "section.description"))
        for question in section.get("question", []):
            q_text = question.get("text", "").strip()
            # PM rule: when section has no explicit sub-question, the section title is
            # repeated as question.text. Skip the duplicate to avoid double-counting.
            if q_text and q_text != sec_title:
                pairs.append((q_text, "question.text"))
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


def evaluate_structured_sample(pred_structured_path: Path, gold_pairs: list[tuple[str, str]]) -> dict:
    """Evaluate a structured DMP JSON prediction against gold pairs.

    Gold-oriented matching: each gold item is matched to the best unused
    predicted item (containment >= 0.75). Unmatched predicted items are
    counted as false positives under the key '__no_gold__' so precision
    is penalised for spurious labels the model invented.
    """
    pred_pairs = extract_gold(pred_structured_path)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    used: set[int] = set()

    for gold_text, gold_label in gold_pairs:
        g_tok = tokenize(gold_text)
        best_s, best_j, best_pl = 0.0, None, None
        for j, (pred_text, pred_label) in enumerate(pred_pairs):
            if j in used:
                continue
            s = containment(tokenize(pred_text), g_tok)
            if s > best_s:
                best_s, best_j, best_pl = s, j, pred_label
        if best_j is not None and best_s >= 0.75:
            used.add(best_j)
            confusion[gold_label][best_pl] += 1
        else:
            confusion[gold_label]["__missed__"] += 1

    # Unmatched predictions = false positives with no gold counterpart
    for j, (_, pred_label) in enumerate(pred_pairs):
        if j not in used:
            confusion["__no_gold__"][pred_label] += 1

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


def micro_prf1(confusion: dict) -> dict:
    """Compute micro-averaged precision/recall/F1 across all labels.

    Unlike ``gold_metrics()`` (recall-only: correct / total gold items),
    this is the standard IE/NER-style metric and also penalizes
    over-generation — predicted blocks with no gold counterpart, stored
    under the ``__no_gold__`` key, count against precision.

    Returns
    -------
    dict with keys: precision, recall, f1, tp, fp, fn
    """
    tp = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
    total_pred = sum(
        v
        for preds in confusion.values()
        for pred_lbl, v in preds.items()
        if pred_lbl != "__missed__"
    )
    total_gold = sum(v for lbl in LABELS for v in confusion.get(lbl, {}).values())

    precision = tp / total_pred if total_pred else 0.0
    recall    = tp / total_gold if total_gold else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": total_pred - tp, "fn": total_gold - tp,
    }


def print_micro_prf1(confusion: dict, label: str = "Overall (micro)") -> None:
    """Print a single, easy-to-read headline line for micro P/R/F1.

    Precision drops when the model over-generates spurious blocks;
    recall drops when it misses gold items; F1 balances both.
    """
    m = micro_prf1(confusion)
    print(
        f"{label:<18}  Precision {m['precision']*100:5.1f}%  "
        f"Recall {m['recall']*100:5.1f}%  F1 {m['f1']*100:5.1f}%  "
        f"(TP={m['tp']}  FP={m['fp']}  FN={m['fn']})"
    )


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
        # FP from wrong-label matches + FP from spurious predictions with no gold counterpart
        fp      = sum(confusion.get(other, {}).get(lbl, 0) for other in LABELS if other != lbl)
        fp     += confusion.get("__no_gold__", {}).get(lbl, 0)
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
    """Extract sample number from a path like sample3_old_dmp.json → 3."""
    return int(path.stem.replace("_old_dmp", "").replace("_dmp", "").replace("sample", ""))


def confusion_matrix_df(confusion: dict):
    """Return a tidy DataFrame of the confusion matrix for display or heatmap plotting.

    Rows are true labels; columns are predicted labels plus ``missed``.
    ``missed`` = gold items with no matching predicted block.
    """
    import pandas as pd
    cols = list(SHORT) + ["missed"]
    data = []
    for true_lbl in LABELS:
        row_dict = confusion.get(true_lbl, {})
        row = [row_dict.get(pred_lbl, 0) for pred_lbl in LABELS]
        row.append(row_dict.get("__missed__", 0))
        data.append(row)
    return pd.DataFrame(data, index=SHORT, columns=cols)


def load_method(tag: str, exclude: list[int] | None = None):
    """Load and evaluate all samples for a given file tag.

    Uses ``sampleN_structured.json`` (DMP template format) as the prediction
    so the comparison against the gold is at the same semantic granularity
    (sections / questions / answers rather than individual PDF lines).

    Parameters
    ----------
    tag:
        Result tag directory name under ``data/output/labeled/``.
    exclude:
        List of sample numbers to skip (e.g. ``[1, 2]`` to skip sample1 and
        sample2 that were used for prompt development).

    Returns
    -------
    tuple[pd.DataFrame, dict, pd.DataFrame] | tuple[None, None, None]
        ``(accuracy_df, confusion_dict, errors_df)`` or ``(None, None, None)``
        if no output files are found.
    """
    import pandas as pd

    _exclude = set(exclude or [])

    samples = sorted(MANUAL_DIR.glob("*_old_dmp.json"), key=_snum)
    def _stem(mp: Path) -> str:
        return mp.stem.replace("_old_dmp", "").replace("_dmp", "")

    found = [mp for mp in samples
             if (LLM_DIR / tag / f"{_stem(mp)}_structured.json").exists()
             and _snum(mp) not in _exclude]
    if not found:
        return None, None, None

    conf_all: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows, errors = [], []

    for mp in samples:
        stem = mp.stem.replace("_old_dmp", "").replace("_dmp", "")
        if _snum(mp) in _exclude:
            continue
        pp = LLM_DIR / tag / f"{stem}_structured.json"
        if not pp.exists():
            continue
        gold = extract_gold(mp)
        conf = evaluate_structured_sample(pp, gold)
        add_confusion(conf_all, conf)

        # Collect mislabeled pairs for the error table (forward-check only).
        for pred_text, pred_label in extract_gold(pp):
            tl = match(pred_text, gold)
            if tl != pred_label:
                errors.append({
                    "sample": stem,
                    "text":   pred_text[:120],
                    "true":   tl or "no_match",
                    "pred":   pred_label,
                })

        correct, _, total = gold_metrics(conf)
        rows.append({
            "sample":   stem,
            "total":    total,
            "correct":  correct,
            "errors":   total - correct,
            "accuracy": correct / total if total else 0,
            "formula":  f"{correct}/{total}",
        })

    return pd.DataFrame(rows), conf_all, pd.DataFrame(errors)


# ── Confidence analysis ───────────────────────────────────────────────────────

def load_confidence(tag: str, exclude: list[int] | None = None):
    """Load predicted blocks with confidence scores for all samples under *tag*.

    Returns a flat DataFrame with one row per predicted block that could be
    matched to a gold label.  Blocks with no gold match are excluded.

    Columns
    -------
    sample, label, confidence, gold_label, correct, page, text
    """
    import pandas as pd

    _exclude = set(exclude or [])
    samples = sorted(MANUAL_DIR.glob("*_old_dmp.json"), key=_snum)
    rows = []
    for mp in samples:
        if _snum(mp) in _exclude:
            continue
        stem = mp.stem.replace("_old_dmp", "").replace("_dmp", "")
        pp   = LLM_DIR / tag / f"{stem}.json"
        if not pp.exists():
            continue
        gold   = extract_gold(mp)
        blocks = json.loads(pp.read_text(encoding="utf-8"))
        for b in blocks:
            gold_label = match(b["text"], gold)
            if gold_label is None:
                continue
            pred_label = b.get("label", "answer.text")
            conf       = float(b.get("confidence", 1.0))
            rows.append({
                "sample":     stem,
                "label":      pred_label,
                "confidence": conf,
                "gold_label": gold_label,
                "correct":    pred_label == gold_label,
                "page":       b.get("page", "-"),
                "text":       b.get("text", "")[:120],
            })
    return pd.DataFrame(rows)


def confidence_calibration_df(conf_df):
    """Bucket blocks by confidence and compute accuracy per bucket.

    Useful for calibration plots: a well-calibrated model has accuracy ≈ confidence
    in each bucket.

    Parameters
    ----------
    conf_df:
        DataFrame from :func:`load_confidence`.

    Returns
    -------
    pd.DataFrame
        Columns: ``bucket_label``, ``mid``, ``count``, ``accuracy``.
    """
    import pandas as pd

    edges  = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    labels = ["<0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–0.95", "≥0.95"]
    mids   = [0.25,   0.55,      0.65,       0.75,       0.85,       0.925,      0.975]

    df = conf_df.copy()
    df["bucket"] = pd.cut(
        df["confidence"], bins=edges, labels=labels, right=False, include_lowest=True
    )
    grouped = (
        df.groupby("bucket", observed=True)
        .agg(count=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .rename(columns={"bucket": "bucket_label"})
    )
    mid_map = dict(zip(labels, mids))
    grouped["mid"] = grouped["bucket_label"].map(mid_map)
    return grouped[["bucket_label", "mid", "count", "accuracy"]]


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
    """Print every gold item for which the prediction has no matching text."""
    pred_pairs = extract_gold(pred_path)
    pred_texts = [t for t, _ in pred_pairs]
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
    """Evaluate one structured DMP JSON against its matching manual annotation."""
    stem        = pred_path.stem.replace("_structured", "").split("_")[0]
    manual_path = MANUAL_DIR / f"{stem}_old_dmp.json"
    if not manual_path.exists():
        logger.warning("No manual label found for %s at %s", stem, manual_path)
        return
    gold      = extract_gold(manual_path)
    confusion = evaluate_structured_sample(pred_path, gold)
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
    print()
    print_micro_prf1(confusion)
    print("\nMissed gold items (LLM produced no matching block):")
    print_missed(pred_path, gold)


def list_tags() -> list[str]:
    """Return all result tags (subdirectories) that have at least one sample JSON."""
    if not LLM_DIR.exists():
        return []
    return sorted(d.name for d in LLM_DIR.iterdir() if d.is_dir() and list(d.glob("sample*.json")))


def run_all(tag: str, exclude: list[int] | None = None) -> None:
    """Evaluate all samples for *tag* against ground truth using structured JSONs."""
    _exclude = set(exclude or [])
    total_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples = sorted(MANUAL_DIR.glob("*_old_dmp.json"), key=_snum)

    excl_note = f"  (excluding sample{list(_exclude)} — used for prompt development)" if _exclude else ""
    print(f"\nEvaluating: {tag}{excl_note}\n")
    _sample_header()
    total_n = total_tp = 0
    per_sample = []

    for manual_path in samples:
        if _snum(manual_path) in _exclude:
            continue
        stem      = manual_path.stem.replace("_old_dmp", "").replace("_dmp", "")
        pred_path = LLM_DIR / tag / f"{stem}_structured.json"
        if not pred_path.exists():
            logger.warning("SKIP %s — no %s", stem, pred_path.name)
            continue
        gold      = extract_gold(manual_path)
        confusion = evaluate_structured_sample(pred_path, gold)
        add_confusion(total_confusion, confusion)

        tp         = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in LABELS)
        n          = sum(sum(v.values()) for v in confusion.values())
        missed_n   = sum(confusion.get(lbl, {}).get("__missed__", 0) for lbl in LABELS)
        mismatch_n = n - tp - missed_n

        if n:
            _sample_row(stem, n, tp)
        else:
            print(f"{stem:<12}  no pairs")
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
    print()
    print_micro_prf1(total_confusion)


def main() -> None:
    """CLI entry point: dmpbridge-evaluate."""
    setup_logging()

    import argparse
    ap = argparse.ArgumentParser(description="Evaluate DMPBridge results against ground truth.")
    ap.add_argument(
        "target", nargs="?",
        help="Result tag (e.g. llama3.1-8b_pdfplumber_whole_doc) or path to a single JSON file.",
    )
    ap.add_argument("--list", "-l", action="store_true", help="List available result tags and exit.")
    ap.add_argument("--exclude", "-x", default="",
                    help="Comma-separated sample numbers to skip, e.g. --exclude 1,2")
    args = ap.parse_args()
    exclude = [int(n) for n in args.exclude.split(",") if n.strip().isdigit()]

    if args.list:
        tags = list_tags()
        if not tags:
            print(f"No results found in {LLM_DIR}")
        else:
            print(f"Available tags in {LLM_DIR}:\n")
            for t in tags:
                n = len(list((LLM_DIR / t).glob("sample*.json")))
                print(f"  {t}  ({n} samples)")
        return

    if args.target and Path(args.target).is_file():
        run_single(Path(args.target))
        return

    if args.target:
        run_all(args.target, exclude=exclude)
        return

    # No argument — auto-detect: use the only tag or prompt the user.
    tags = list_tags()
    if not tags:
        print(f"No results found in {LLM_DIR}. Run an experiment first.")
        sys.exit(1)
    if len(tags) == 1:
        run_all(tags[0], exclude=exclude)
    else:
        print("Multiple result sets found. Specify one with:\n")
        for t in tags:
            print(f"  dmpbridge-evaluate {t}")
        sys.exit(1)


if __name__ == "__main__":
    main()
