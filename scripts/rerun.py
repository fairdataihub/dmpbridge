"""Re-run configurations safely — back up, clear the cache, run, score.

The pipeline has no --force flag: stages 2-4 are cached by file existence, so a
completed tag is silently skipped. Clearing those by hand is easy to get wrong in
two ways — deleting stage 1 as well (throwing away extraction shared by every
model), or deleting results with no copy kept.

    python scripts/rerun.py --model llama3.1:8b
    python scripts/rerun.py --all                       # all three models
    python scripts/rerun.py --all --extractor docling
    python scripts/rerun.py --model gemma4:e4b --dry-run

Stage 1 is never touched. Existing stages 2-4 are copied under
Report-doc/backups/<timestamp>/ before deletion.

llama3.3:70b takes ~20 minutes for 10 documents. Run this in the background if
the 70B is included.
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dmpbridge.core import paths as P

MODELS = ["llama3.1:8b", "gemma4:e4b", "llama3.3:70b"]
STAGES = [("2_labeled", P.LABELED_DIR),
          ("3_structured", P.STRUCTURED_DIR),
          ("4_final", P.FINAL_DIR)]
BACKUP_ROOT = Path("Report-doc/backups")


def tag_for(model: str, extractor: str) -> str:
    """Mirror the tag the CLI writes, without importing its argument parser."""
    return f"{model.replace(':', '-')}_{extractor}_whole_doc"


def back_up(tag: str, stamp: str) -> int:
    """Copy existing stages 2-4 aside. Returns how many stages were saved."""
    dest = BACKUP_ROOT / stamp / tag
    saved = 0
    for name, root in STAGES:
        src = root / tag
        if src.exists():
            shutil.copytree(src, dest / name, dirs_exist_ok=True)
            saved += 1
    return saved


def clear(tag: str) -> None:
    """Delete stages 2-4 for one tag. Stage 1 is deliberately left alone."""
    for _, root in STAGES:
        shutil.rmtree(root / tag, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", action="append", help="model to re-run (repeatable)")
    g.add_argument("--all", action="store_true", help=f"re-run {', '.join(MODELS)}")
    ap.add_argument("--extractor", default="pdfplumber",
                    choices=["pdfplumber", "docling", "lighton"])
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--no-backup", action="store_true", help="skip the backup copy")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen")
    args = ap.parse_args()

    models = MODELS if args.all else args.model
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print(f"extractor : {args.extractor}")
    print(f"samples   : {args.start}-{args.end}")
    print(f"models    : {', '.join(models)}")
    if not args.no_backup and not args.dry_run:
        print(f"backup    : {BACKUP_ROOT / stamp}")
    print()

    for model in models:
        tag = tag_for(model, args.extractor)
        existing = sum((root / tag).exists() for _, root in STAGES)

        if args.dry_run:
            print(f"[{model}] would clear {existing} stage(s) of {tag}, then run")
            continue

        if existing and not args.no_backup:
            n = back_up(tag, stamp)
            print(f"[{model}] backed up {n} stage(s)")
        clear(tag)

        cmd = ["dmpbridge-wholedoc", "--model", model, "--extractor", args.extractor,
               "--start", str(args.start), "--end", str(args.end)]
        print(f"[{model}] {' '.join(cmd)}")
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            # Stop rather than continue: a failed run leaves a partial tag, and
            # scoring a partial tag silently produces a wrong number.
            print(f"[{model}] FAILED (exit {r.returncode}) — stopping. "
                  f"Partial output left in place; restore from {BACKUP_ROOT / stamp} "
                  f"or re-run this model.")
            sys.exit(r.returncode)

    if not args.dry_run:
        print("\n" + "=" * 60)
        subprocess.run([sys.executable, "scripts/compare_results.py", "--questions"],
                       check=False)


if __name__ == "__main__":
    main()
