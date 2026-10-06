"""Reference-free check of RDA maDMP JSON outputs: is each value the model wrote
actually in the PDF's text?

Only 3 of the pilot samples have a hand-made reference, so precision/recall cannot
be computed for the rest. This script needs no reference. For every free-text value
a model wrote (titles, descriptions, names, identifiers, URLs, hosts, licenses) it
tests whether the value appears in the pdfplumber text of that sample:

  grounded     the value, or at least 80% of its words, is in the plan's text
  placeholder  "unknown", "not specified", "N/A", "none" and similar - not a value
  invented     anything else: nothing in the plan supports it

Fields the schema restricts to fixed values (type, data_access, personal_data,
language, ...) and the dates the schema forces (created, modified) are not checked,
because the model has no free choice there.

    python scripts/rda_grounding_check.py v6            # every sample in runs/v6
    python scripts/rda_grounding_check.py v2 v6 --samples 3 10 14
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from dmpbridge.extractors import get_extractor

RUNS = Path("data/output/rda/runs")
PDFS = Path("data/input/pdfs")
NOT_CHECKED = {"type", "data_access", "personal_data", "sensitive_data", "ethical_issues_exist", "language",
               "funding_status", "is_reused", "certified_with", "pid_system", "support_versioning", "geo_location",
               "storage_type", "availability", "backup_type", "backup_frequency", "currency_code", "role",
               "created", "modified", "issued", "start", "end", "start_date", "available_until"}
PLACEHOLDERS = {"unknown", "not specified", "n/a", "na", "none", "not applicable", "not provided", "null",
                "not available", "tbd", "unspecified", "not stated", "not mentioned"}

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--samples", nargs="*", type=int)
ap.add_argument("--show", type=int, default=0, help="print up to N invented values per model")
ap.add_argument("--dir", default=str(RUNS), help="folder that holds the run folders")
args = ap.parse_args()
RUNS = Path(args.dir)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def words(s):
    return [w for w in norm(s).split() if len(w) > 2]


def leaves(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, node


texts = {}


def plan_text(n):
    if n not in texts:
        t = get_extractor("pdfplumber").extract(PDFS / f"sample{n}.pdf")[0]["text"]
        texts[n] = (norm(t), set(words(t)))
    return texts[n]


def judge(value, n):
    if value is None or str(value).strip() == "":
        return "empty"
    if norm(value) in PLACEHOLDERS:
        return "placeholder"
    flat, vocab = plan_text(n)
    v = norm(value)
    if v and v in flat:
        return "grounded"
    w = words(value)
    if w and sum(x in vocab for x in w) / len(w) >= 0.8:
        return "grounded"
    return "invented"


rows = defaultdict(lambda: defaultdict(int))
examples = defaultdict(list)
for run in args.runs:
    for f in sorted((RUNS / run).glob("sample*.rda.*.json")):
        m = re.match(r"sample(\d+)\.rda\.(.+)\.json$", f.name)
        n, model = int(m[1]), m[2]
        if args.samples and n not in args.samples:
            continue
        for path, value in leaves(json.loads(f.read_text(encoding="utf-8"))):
            field = re.sub(r"\[\d+\]", "", path.split(".")[-1])
            if field in NOT_CHECKED or isinstance(value, bool) or isinstance(value, (int, float)):
                continue
            verdict = judge(value, n)
            rows[(run, model)][verdict] += 1
            rows[(run, model)]["samples"] = rows[(run, model)].get("samples", 0)
            if verdict == "invented":
                examples[(run, model)].append(f"sample{n} {path} = {str(value)[:70]!r}")
        for key in [(run, model)]:
            rows[key].setdefault("_samples", set()).add(n)

print(f"{'run':12} {'model':14} {'samples':>7} {'grounded':>9} {'invented':>9} {'placeholder':>12} {'empty':>6}   share grounded")
for (run, model), r in sorted(rows.items()):
    written = r["grounded"] + r["invented"] + r["placeholder"]
    share = r["grounded"] / written if written else 0
    print(f"{run:12} {model:14} {len(r['_samples']):7} {r['grounded']:9} {r['invented']:9} {r['placeholder']:12} {r['empty']:6}   {share:.0%}")
    for e in examples[(run, model)][:args.show]:
        print(f"{'':28}invented: {e}")
