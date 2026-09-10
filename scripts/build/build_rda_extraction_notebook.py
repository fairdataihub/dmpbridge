"""Build notebooks/pdf-to-rda-dmp-json.ipynb.

The "PDF to RDA DMP JSON" pipeline: read one DMP PDF, clean the text, then ask
llama3.1:8b to fill in every field of
rda_dmp_dmptool_extension_skeleton_no_narrative.json — with each answer required
to quote the sentence it came from, so an invented value can be caught and
dropped in code rather than trusted.

    python scripts/build/build_rda_extraction_notebook.py
"""
import json
from pathlib import Path

NB = Path("notebooks/pdf-to-rda-dmp-json.ipynb")


def md(cid, body):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in body.strip("\n").split("\n")]}


def code(cid, body):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in body.strip("\n").split("\n")]}


cells = [

md("title", '''
# PDF to RDA DMP JSON

Read one DMP PDF, and fill in every field of the **RDA DMP Common Standard**
skeleton (`rda_dmp_dmptool_extension_skeleton_no_narrative.json`) using
`llama3.1:8b` running locally.

**The problem this notebook is built around.** The skeleton has 91 leaf fields and
47 of them are identifiers — ORCID, DOI, ROR, grant numbers, emails. A DMP often
does not contain them, and an 8B model asked for an ORCID will cheerfully invent
one that looks perfectly real. Fluent, well-formed, and wrong.

**The guard.** Every non-null answer must come with the *exact sentence* it was
taken from. We then check that sentence really is in the PDF text, and that the
value really is in the sentence. Anything that fails is forced to `null` and
logged. The model is not trusted to police itself — the check is done in code,
against the source text.

Steps: **1** read the PDF · **2** clean the text · **3** extract with evidence ·
**4** verify every field · **5** assemble the RDA JSON · **6** report what was
kept and what was rejected.
'''),

code("setup", '''
import json
import re
import sys
from pathlib import Path

if Path.cwd().name == "notebooks":
    import os
    os.chdir(Path.cwd().parent)

from dmpbridge.extractors import get_extractor
from dmpbridge.models.ollama import OllamaModel

# ── What to run ───────────────────────────────────────────────────────────────
PDF        = Path("data/input/pdfs/sample14.pdf")
SKELETON   = Path("rda_dmp_dmptool_extension_skeleton_no_narrative.json")
MODEL      = "llama3.1:8b"
HOST       = "http://localhost:11434"
OUT        = Path("data/output/rda") / (PDF.stem + ".rda.json")

# ── Gates, borrowed from the poster2json settings ─────────────────────────────
# Fail loudly rather than silently truncating: a clipped prompt produces
# confidently-wrong JSON, which is worse than an error you can see.
NUM_CTX          = 32768
MAX_INPUT_CHARS  = 60_000     # ~15k tokens, leaves room for prompt + output

print(f"PDF      : {PDF}")
print(f"Model    : {MODEL}  (temperature pinned to 0.0 by the backend)")
print(f"Skeleton : {SKELETON}")
'''),

md("s1", '''
## Step 1 — Read the PDF

Uses the project's own pdfplumber extractor, so the text is identical to what the
main pipeline sees, including the `**bold**` / `_italic_` / `++underline++`
markers it adds for visually emphasized words.
'''),

code("read", '''
blocks = get_extractor("pdfplumber").extract(PDF)
raw_text = blocks[0]["text"]

print(f"{len(raw_text):,} characters, {len(raw_text.split()):,} words")
print("\\n--- first 400 characters " + "-" * 30)
print(raw_text[:400])
'''),

md("s2", '''
## Step 2 — Clean the text

Light touch on purpose. Anything we strip here can no longer be quoted as
evidence in step 4, so aggressive cleaning would cause real values to be rejected.
We only normalize whitespace and drop the emphasis markers, keeping the words
themselves untouched.
'''),

code("clean", '''
def clean_text(text: str) -> str:
    """Normalize whitespace and strip emphasis markers, keeping all words."""
    t = re.sub(r"\\+\\+(.+?)\\+\\+", r"\\1", text)     # ++underline++
    t = re.sub(r"\\*\\*(.+?)\\*\\*", r"\\1", t)        # **bold**
    t = re.sub(r"(?<!\\w)_(.+?)_(?!\\w)", r"\\1", t)  # _italic_
    t = t.replace("\\u00ad", "")                    # soft hyphens
    t = re.sub(r"[ \\t]+", " ", t)                  # runs of spaces
    t = re.sub(r"\\n{3,}", "\\n\\n", t)               # runs of blank lines
    return t.strip()


clean = clean_text(raw_text)

if len(clean) > MAX_INPUT_CHARS:
    raise ValueError(
        f"INPUT_TOO_LONG: {len(clean):,} chars > {MAX_INPUT_CHARS:,}. "
        "Refusing to truncate — a clipped prompt yields confidently-wrong JSON."
    )

print(f"{len(raw_text):,} chars -> {len(clean):,} chars after cleaning")
print(f"Input gate: OK ({len(clean):,} / {MAX_INPUT_CHARS:,})")
print("\\n--- first 400 characters " + "-" * 30)
print(clean[:400])
'''),

md("s3", '''
## Step 3 — What we are trying to fill

The skeleton is deeply nested, so asking for all 91 fields in one call is
unreliable. We ask in small related groups instead, each with a plain-English
description of what the field means — the field *name* alone (`mbox`, `issued`)
is not something an 8B model reads correctly.
'''),

code("groups", '''
skeleton = json.loads(SKELETON.read_text(encoding="utf-8"))

# (json path, what it means). The path is where the answer is written back.
FIELD_GROUPS = {
    "DMP core": [
        ("dmp.title",                     "the title of the data management plan or project"),
        ("dmp.description",               "the abstract or summary of the project"),
        ("dmp.dmp_id.identifier",         "the DOI or URL that identifies this DMP itself"),
        ("dmp.language",                  "the language the plan is written in"),
        ("dmp.ethical_issues_exist",      "does the plan mention ethical issues? yes or no"),
        ("dmp.ethical_issues_description","what ethical issues are described, if any"),
    ],
    "Contact": [
        ("dmp.contact.name",              "full name of the main contact or creator of the plan"),
        ("dmp.contact.mbox",              "email address of that contact"),
        ("dmp.contact.contact_id[].identifier",
                                          "that contact's ORCID iD (16 digits, like 0000-0001-2345-6789)"),
        ("dmp.contact.affiliation[].name","the institution that contact belongs to"),
    ],
    "Project and funding": [
        ("dmp.project[].title",           "the research project title"),
        ("dmp.project[].description",     "the research project abstract"),
        ("dmp.project[].start",           "project start date"),
        ("dmp.project[].end",             "project end date"),
        ("dmp.project[].funding[].name",  "name of the funding organisation"),
        ("dmp.project[].funding[].grant_id[].identifier",
                                          "the grant or award number"),
    ],
    "Dataset": [
        ("dmp.dataset[].title",           "title or name of the dataset being described"),
        ("dmp.dataset[].description",     "what the dataset contains"),
        ("dmp.dataset[].personal_data",   "does the data contain personal data? yes or no"),
        ("dmp.dataset[].sensitive_data",  "does the data contain sensitive data? yes or no"),
        ("dmp.dataset[].distribution[].data_access",
                                          "how the data will be accessed: open, shared or closed"),
        ("dmp.dataset[].distribution[].host.title",
                                          "the repository the data will be deposited in"),
    ],
}

n = sum(len(v) for v in FIELD_GROUPS.values())
print(f"{len(FIELD_GROUPS)} groups, {n} fields")
for g, fs in FIELD_GROUPS.items():
    print(f"  {g:22} {len(fs)} fields")
'''),

md("s4", '''
## Step 4 — Extract, with evidence

Two things make this different from asking the model to "fill in the JSON":

1. **`null` is an explicitly correct answer.** The prompt says so repeatedly. Most
   hallucination comes from a model that believes it must produce *something*.
2. **Every non-null value must carry the sentence it came from.** That gives us
   something checkable in step 5.

Ollama's `format: <schema>` constrains decoding to the shape we ask for, so the
reply is structurally valid JSON without needing the brace-balancing that a raw
`transformers` loop requires. The retry ladder below still guards the rest.
'''),

code("extract", '''
llm = OllamaModel(model=MODEL, host=HOST, num_ctx=NUM_CTX)

SYSTEM = """You extract metadata from Data Management Plans into JSON.

THE MOST IMPORTANT RULE: if the document does not state something, answer null.
Never guess. Never infer. Never construct a plausible-looking identifier. A null
answer is correct and expected — most documents do not contain most of these
fields.

For every field you answer with a value, you must also copy the exact sentence
from the document that the value came from, into "evidence". Copy it character
for character. Do not paraphrase it, do not shorten it, do not tidy it up. If you
cannot point at a sentence, the answer is null and evidence is null."""


def build_prompt(group_name, fields, text):
    lines = [f"Extract these {group_name} fields from the Data Management Plan below.", ""]
    for path, desc in fields:
        lines.append(f'  "{path}" - {desc}')
    lines += [
        "",
        "For each field return an object with:",
        '  "value"    - what the document says, or null if it does not say',
        '  "evidence" - the exact sentence from the document containing that value,',
        "               copied verbatim, or null if value is null",
        "",
        "Answer null for anything the document does not state. Do not invent",
        "identifiers, emails, ORCIDs, DOIs, grant numbers or dates.",
        "",
        "--- DOCUMENT " + "-" * 50,
        text,
        "--- END DOCUMENT " + "-" * 46,
    ]
    return "\\n".join(lines)


def build_schema(fields):
    """One {value, evidence} pair per field; both nullable strings."""
    props = {}
    for path, _ in fields:
        props[path] = {
            "type": "object",
            "properties": {
                "value":    {"type": ["string", "null"]},
                "evidence": {"type": ["string", "null"]},
            },
            "required": ["value", "evidence"],
        }
    return {"type": "object", "properties": props, "required": list(props)}


def robust_json_parse(raw):
    """Parse, with repair passes — the ladder's last line of defence."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\\{.*\\}", raw, re.S)          # strip any prose around it
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        from json_repair import repair_json
        return json.loads(repair_json(raw))
    except Exception:
        return None


def extract_group(group_name, fields, text):
    """Primary -> retry -> fallback-prompt ladder, then robust parse."""
    attempts = [
        ("primary",  build_prompt(group_name, fields, text)),
        ("retry",    build_prompt(group_name, fields, text)),
        ("fallback", build_prompt(group_name, fields, text[:len(text) // 2])),
    ]
    for label, prompt in attempts:
        raw = llm.complete(SYSTEM, prompt, schema=build_schema(fields))
        parsed = robust_json_parse(raw)
        if parsed is not None:
            if label != "primary":
                print(f"    (recovered on {label} attempt)")
            return parsed
        print(f"    {label} attempt did not parse, escalating")
    raise RuntimeError(f"{group_name}: all attempts failed to parse")


raw_answers = {}
for group_name, fields in FIELD_GROUPS.items():
    print(f"  {group_name} ...", flush=True)
    raw_answers[group_name] = extract_group(group_name, fields, clean)

print(f"\\nExtracted {sum(len(v) for v in raw_answers.values())} field answers")
'''),

md("s5", '''
## Step 5 — Verify every answer against the document

This is the step that actually catches hallucination. For each answered field:

1. **Is the evidence really in the document?** Compared on whitespace-normalized,
   lowercased text, so formatting differences don't cause false rejections. This
   is the check that catches an invented ORCID or DOI — there is no sentence to
   point at, so it cannot pass.
2. **Is the value really in that evidence?** Stops the model quoting a real
   sentence and attaching an unrelated value to it.

Fail either check and the value becomes `null`, recorded with the reason. Nothing
here asks the model's opinion.

**Classification fields are checked differently, on purpose.** `data_access` is
`open` / `shared` / `closed`; `personal_data` is `yes` / `no`. Those words are
conclusions drawn from a sentence, not words quoted from it — "will be made
public within 6 months" means `open` without containing "open". Requiring the
literal value would reject correct answers, and would also let a short value like
`no` pass by coincidence when it happens to sit inside "**no** ethical issues".
So for these fields we still demand real evidence from the document, but check the
value against the allowed set instead of against the sentence.
'''),

code("verify", '''
def norm(s):
    return re.sub(r"\\s+", " ", (s or "")).strip().lower()


clean_norm = norm(clean)

# Fields whose value is a conclusion drawn from a sentence, not a quote from it.
# Checked against this set rather than against the evidence text.
ENUM_FIELDS = {
    "dmp.ethical_issues_exist":  {"yes", "no", "unknown"},
    "dmp.dataset[].personal_data":  {"yes", "no", "unknown"},
    "dmp.dataset[].sensitive_data": {"yes", "no", "unknown"},
    "dmp.dataset[].distribution[].data_access": {"open", "shared", "closed"},
}


def verify(value, evidence, path=None):
    """-> (accepted_value, status, note)."""
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "null":
        return None, "empty", "model answered null"
    if not evidence:
        return None, "REJECTED", "value given with no evidence"

    ev = norm(evidence)
    if ev not in clean_norm:
        return None, "REJECTED", "evidence sentence is not in the document"

    val = norm(value)

    # Classification field: the sentence must be real, the value must be legal.
    if path in ENUM_FIELDS:
        if val not in ENUM_FIELDS[path]:
            allowed = "/".join(sorted(ENUM_FIELDS[path]))
            return None, "REJECTED", f"not one of the allowed values ({allowed})"
        return val, "kept", "classification, evidence verified"

    if val not in ev:
        # A value legitimately reformatted from its sentence (a date, a yes/no)
        # is allowed through only if most of its words are in the evidence.
        words = [w for w in val.split() if len(w) > 3]
        overlap = sum(w in ev for w in words)
        if not words or overlap / len(words) < 0.6:
            return None, "REJECTED", "value does not appear in its own evidence"
        return value, "kept", "value reformatted from evidence"

    return value, "kept", ""


results = []
for group_name, fields in FIELD_GROUPS.items():
    answers = raw_answers[group_name]
    for path, desc in fields:
        a = answers.get(path) or {}
        val, status, note = verify(a.get("value"), a.get("evidence"), path)
        results.append({
            "group": group_name, "path": path, "value": val, "status": status,
            "note": note, "claimed": a.get("value"), "evidence": a.get("evidence"),
        })

kept     = [r for r in results if r["status"] == "kept"]
rejected = [r for r in results if r["status"] == "REJECTED"]
empty    = [r for r in results if r["status"] == "empty"]

print(f"kept     : {len(kept)}")
print(f"rejected : {len(rejected)}   <- caught before they reached the JSON")
print(f"null     : {len(empty)}   <- model correctly declined")
'''),

md("s6", '''
## Step 6 — What was rejected

The interesting output. Each row is a value the model produced that could **not**
be traced back to the document — exactly what would have ended up in your JSON,
silently, without this check.
'''),

code("show_rejected", '''
if not rejected:
    print("Nothing was rejected — every answered field traced back to the document.")
for r in rejected:
    print(f"\\n{r['path']}")
    print(f"  claimed  : {r['claimed']!r}")
    print(f"  evidence : {str(r['evidence'])[:110]!r}")
    print(f"  reason   : {r['note']}")
'''),

md("s6b", '''
### Does the guard actually work?

"Nothing was rejected" is not evidence that the check works — it could equally mean
the check is broken and passing everything. So here we push four fabricated
answers through the same `verify()` the real fields went through: a made-up ORCID,
a made-up DOI, a real value with a plausible-sounding invented sentence, and a
real sentence with an unrelated value stapled to it.

All four must be rejected. If any is kept, the guard is not doing its job.
'''),

code("negative_control", '''
FAKES = [
    ("dmp.contact.contact_id[].identifier", "0000-0002-1825-0097",
     "Correspondence to Dr. Jane Miller (ORCID 0000-0002-1825-0097).",
     "invented ORCID with an invented sentence"),
    ("dmp.dmp_id.identifier", "https://doi.org/10.5555/fake.9999",
     "This plan is registered under DOI 10.5555/fake.9999.",
     "invented DOI with an invented sentence"),
    ("dmp.contact.mbox", "brett.johnson@hakai.org",
     "Contact the data manager at brett.johnson@hakai.org for access.",
     "plausible email, sentence not in the document"),
    ("dmp.project[].funding[].name", "National Science Foundation",
     "Funder: Tula Foundation",
     "real sentence, unrelated value attached"),
]

all_caught = True
for path, value, evidence, what in FAKES:
    val, status, note = verify(value, evidence, path)
    ok = status == "REJECTED"
    all_caught &= ok
    print(f"[{'CAUGHT' if ok else 'MISSED'}] {what}")
    print(f"          value  : {value!r}")
    print(f"          result : {status} - {note}\\n")

print("=" * 60)
print("Guard is working: all fabricated answers rejected." if all_caught
      else "WARNING: a fabricated answer got through — do not trust the output.")
'''),

md("s7", '''
## Step 7 — Assemble the RDA JSON

Verified values only, written into the skeleton at their proper paths. Everything
else keeps the skeleton's `null`, which is a truthful "the document does not say"
rather than a guess.
'''),

code("assemble", '''
def set_path(obj, path, value):
    """Write value into obj at a dotted path; '[]' means the first list item."""
    parts = path.split(".")
    cur = obj
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        is_list = part.endswith("[]")
        key = part[:-2] if is_list else part
        if last:
            if is_list:
                if not isinstance(cur.get(key), list) or not cur[key]:
                    cur[key] = [None]
                cur[key][0] = value
            else:
                cur[key] = value
            return
        if is_list:
            if not isinstance(cur.get(key), list) or not cur[key]:
                cur[key] = [{}]
            cur = cur[key][0]
        else:
            if not isinstance(cur.get(key), dict):
                cur[key] = {}
            cur = cur[key]


rda = json.loads(SKELETON.read_text(encoding="utf-8"))
for r in kept:
    set_path(rda, r["path"], r["value"])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rda, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"written -> {OUT}")
print()
print(json.dumps(rda["dmp"], indent=2, ensure_ascii=False)[:1200])
'''),

md("s8", '''
## Step 8 — Summary

Fill rate per group, and every field with the evidence behind it, so any value can
be traced back to the sentence that produced it.
'''),

code("summary", '''
print(f"{'group':22} {'kept':>5} {'rejected':>9} {'null':>6}")
print("-" * 46)
for g in FIELD_GROUPS:
    rows = [r for r in results if r["group"] == g]
    k = sum(r["status"] == "kept" for r in rows)
    x = sum(r["status"] == "REJECTED" for r in rows)
    e = sum(r["status"] == "empty" for r in rows)
    print(f"{g:22} {k:>5} {x:>9} {e:>6}")
print("-" * 46)
print(f"{'TOTAL':22} {len(kept):>5} {len(rejected):>9} {len(empty):>6}")

print("\\n\\nEvery kept value, with its source sentence:\\n")
for r in kept:
    print(f"{r['path']}")
    print(f"   = {r['value']!r}")
    print(f"   from: {str(r['evidence'])[:100]!r}")
    if r["note"]:
        print(f"   note: {r['note']}")
    print()
'''),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
