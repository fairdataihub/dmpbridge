"""Build notebooks/investigation-lightonocr-native-output-sample2.ipynb.

Companion to build_docling_native_notebook.py, same question for LightOnOCR:
does its native output carry bold, italic or underline? LightOnOCR has no
document JSON and no page cells — it is a vision-language model whose only
output is generated text — so "native" here means what the model writes on
its own, with a plain transcription prompt and no marker instructions. That
is compared with what it writes under the project's marker prompt, and both
are measured against pdfplumber's font-derived 42 bold / 11 underline /
41 italic runs on sample 2.

Needs the CUDA GPU and the dmpbridge[lighton] extras; ~3 minutes to execute.

    python scripts/build/build_lightonocr_native_notebook.py
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \\
        notebooks/investigation-lightonocr-native-output-sample2.ipynb
"""
import json
from pathlib import Path

NB = Path("notebooks/investigation-lightonocr-native-output-sample2.ipynb")
SAMPLE = 2


def md(cid, lines):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(cid, lines):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = [
    md("title", [
        f"# Does LightOnOCR's native output capture bold, italic, underline? — sample {SAMPLE}",
        "",
        "Companion to `investigation-docling-native-json-sample2.ipynb`. Docling had three",
        "levels to look at (JSON, Markdown, page cells). LightOnOCR has **one**: it is a",
        "vision-language model that looks at a picture of the page and writes text, and that",
        "text is the whole output. There is no data structure underneath it to inspect.",
        "",
        "So the question becomes: what formatting does the model write **on its own**, and",
        "what does it write **when asked** for the project's markers? Two runs on the same",
        "page images:",
        "",
        "| run | prompt | what it shows |",
        "|---|---|---|",
        "| native | \"Transcribe all text on this page exactly as it appears, in reading order.\" — no mention of formatting | the model's own habits |",
        "| project | the extractor's `_TRANSCRIBE_PROMPT`, which asks for `**bold**`, `_italic_`, `++underline++` | what the pipeline actually gets |",
        "",
        f"Sample {SAMPLE} is the test document because pdfplumber, reading the PDF's font data,",
        "finds **42 bold, 11 underlined and 41 italic** runs in it. Decoding is greedy, so both",
        "runs are deterministic and the project run should reproduce the cached stage-1 text.",
    ]),

    code("setup", [
        "import os",
        "from pathlib import Path",
        "",
        "if Path.cwd().name == 'notebooks':",
        "    os.chdir(Path.cwd().parent)",
        "",
        "import collections",
        "import json",
        "import logging",
        "import re",
        "import time",
        "import warnings",
        "",
        "os.environ['TQDM_DISABLE'] = '1'",
        "warnings.filterwarnings('ignore')",
        "logging.disable(logging.WARNING)",
        "",
        "import pandas as pd",
        "import torch",
        "from IPython.display import display",
        "",
        "pd.set_option('display.max_colwidth', 80)",
        "",
        f"SAMPLE = {SAMPLE}",
        "PDF = Path(f'data/input/pdfs/sample{SAMPLE}.pdf')",
        "",
        "from dmpbridge.extractors import get_extractor",
        "from dmpbridge.extractors import lighton_extractor as L",
        "",
        "t0 = time.time()",
        "ex = get_extractor('lightonocr')",
        "print(f'model {L.DEFAULT_MODEL_ID} loaded in {time.time() - t0:.0f} s, '",
        "      f'{torch.cuda.memory_allocated() / 1024**3:.1f} GB VRAM')",
        "",
        "pages = ex._render_pages(PDF)",
        "print(f'{len(pages)} pages rendered at {L._RENDER_DPI} dpi, {pages[0].size[0]}x{pages[0].size[1]} px')",
        "",
        "",
        "def ocr(image, prompt):",
        "    \"\"\"The extractor's _ocr_image, with the prompt as a parameter.\"\"\"",
        "    messages = [{'role': 'user', 'content': [{'type': 'image'},",
        "                                             {'type': 'text', 'text': prompt}]}]",
        "    text_prompt = ex._processor.apply_chat_template(messages, add_generation_prompt=True)",
        "    inputs = ex._processor(text=text_prompt, images=image, return_tensors='pt').to(ex._device)",
        "    with torch.no_grad():",
        "        out = ex._model.generate(**inputs, max_new_tokens=ex._max_new_tokens)",
        "    return ex._processor.decode(out[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)",
        "",
        "",
        "NATIVE_PROMPT = ('Transcribe all text on this document page exactly as it appears, '",
        "                 'in reading order.')",
        "PROJECT_PROMPT = L._TRANSCRIBE_PROMPT",
        "print()",
        "print('project prompt:', PROJECT_PROMPT)",
    ]),

    code("run", [
        "runs = {}",
        "for name, prompt in (('native', NATIVE_PROMPT), ('project', PROJECT_PROMPT)):",
        "    t0 = time.time()",
        "    runs[name] = '\\n\\n'.join(ocr(img, prompt) for img in pages)",
        "    print(f'{name:8s} {len(runs[name]):6d} chars   {time.time() - t0:5.1f} s')",
        "",
        "cached = json.loads(Path(f'data/output/1_extracted/lightonocr/sample{SAMPLE}.json')",
        "                    .read_text(encoding='utf-8'))[0]['text']",
        "print()",
        "print('project run reproduces cached stage 1 exactly:', runs['project'] == cached)",
    ]),

    # ── 1 ─────────────────────────────────────────────────────────────────
    md("md-1", [
        "## 1. What markup each run contains",
        "",
        "Counted directly in the generated text. `#` headings and `**bold**` are the model's",
        "two ways of showing emphasis; `*italic*` is Markdown's italic; `_italic_` and",
        "`++underline++` are the project's requested forms.",
    ]),
    code("counts", [
        "def markup(t):",
        "    return {",
        "        '# headings':      len(re.findall(r'^#+ ', t, re.M)),",
        "        '**bold**':        len(re.findall(r'\\*\\*[^*\\n]+?\\*\\*', t)),",
        "        '*italic*':        len(re.findall(r'(?<![*\\w])\\*(?!\\*)[^*\\n]+?\\*(?![*\\w])', t)),",
        "        '_italic_':        len(re.findall(r'(?<!\\w)_[^_\\n]+?_(?!\\w)', t)),",
        "        '++underline++':   len(re.findall(r'\\+\\+[^+\\n]+?\\+\\+', t)),",
        "        '<u>':             t.count('<u>'),",
        "    }",
        "",
        "table = pd.DataFrame({name: markup(t) for name, t in runs.items()})",
        "table['PDF has (pdfplumber)'] = pd.Series({'**bold**': 42, '_italic_': 41, '++underline++': 11})",
        "display(table.fillna('').astype(str))",
    ]),
    md("md-1b", [
        "The opening of each run, so the difference is visible rather than counted.",
    ]),
    code("heads", [
        "for name, t in runs.items():",
        "    print(f'───── {name} ─────')",
        "    print('\\n'.join(t.splitlines()[:14]))",
        "    print()",
    ]),

    # ── 2 ─────────────────────────────────────────────────────────────────
    md("md-2", [
        "## 2. Reference — the runs the PDF actually contains",
        "",
        "pdfplumber's stage-1 text for the same document, with its font-derived markers.",
        "Each of its bold / underline / italic runs is then looked up in both LightOnOCR",
        "outputs to see how the model rendered that exact text.",
    ]),
    code("reference", [
        "stage1 = json.loads(Path(f'data/output/1_extracted/pdfplumber/sample{SAMPLE}.json')",
        "                    .read_text(encoding='utf-8'))[0]['text']",
        "ref = {",
        "    'bold':      re.findall(r'\\*\\* (.+?) \\*\\*', stage1),",
        "    'underline': re.findall(r'\\+\\+ (.+?) \\+\\+', stage1),",
        "    'italic':    re.findall(r'(?<![\\w*+])_ (.+?) _(?![\\w])', stage1),",
        "}",
        "print({k: len(v) for k, v in ref.items()})",
        "",
        "",
        "def clean(s):",
        "    return re.sub(r'\\s+', ' ', re.sub(r'[*_+#]', '', s)).strip()",
        "",
        "",
        "def rendered_as(text, phrase):",
        "    \"\"\"Find *phrase* in *text* and report which markers surround it on its line.\"\"\"",
        "    key = clean(phrase)[:28]",
        "    for line in text.splitlines():",
        "        if key in clean(line):",
        "            if re.match(r'^#+ ', line):",
        "                return 'heading ' + line.split()[0]",
        "            i = clean(line).find(key)",
        "            tags = []",
        "            if re.search(r'\\*\\*[^*]*' + re.escape(key[:12]), line):  tags.append('**bold**')",
        "            if re.search(r'(?<!\\*)\\*(?!\\*)[^*]*' + re.escape(key[:12]), line): tags.append('*italic*')",
        "            if re.search(r'(?<!\\w)_[^_]*' + re.escape(key[:12]), line): tags.append('_italic_')",
        "            if '++' in line: tags.append('++underline++')",
        "            return ', '.join(tags) or 'plain'",
        "    return 'not found'",
        "",
        "",
        "rows = []",
        "for kind, phrases in ref.items():",
        "    for p in phrases:",
        "        rows.append({'signal in PDF': kind, 'text': p[:60],",
        "                     'native run': rendered_as(runs['native'], p),",
        "                     'project run': rendered_as(runs['project'], p)})",
        "per_run = pd.DataFrame(rows)",
        "display(per_run.head(24))",
    ]),
    md("md-2b", [
        "Summed over every run in the PDF: how many did each LightOnOCR output mark **at all**",
        "(with any marker), and how many with the **right** marker.",
    ]),
    code("recall", [
        "RIGHT = {'bold': ('**bold**', 'heading'), 'italic': ('*italic*', '_italic_'),",
        "         'underline': ('++underline++',)}",
        "",
        "summary = []",
        "for kind in ref:",
        "    sub = per_run[per_run['signal in PDF'] == kind]",
        "    for run in ('native run', 'project run'):",
        "        marked = sub[run].apply(lambda v: v not in ('plain', 'not found'))",
        "        right  = sub[run].apply(lambda v: any(r in v for r in RIGHT[kind]))",
        "        summary.append({'signal': kind, 'run': run.split()[0], 'in PDF': len(sub),",
        "                        'marked at all': int(marked.sum()),",
        "                        'marked as ' + '/'.join(RIGHT[kind]): int(right.sum()),",
        "                        'not found in output': int((sub[run] == 'not found').sum())})",
        "display(pd.DataFrame(summary).fillna('').astype(str))",
    ]),

    # ── 3 ─────────────────────────────────────────────────────────────────
    md("md-3", [
        "## 3. Things the model wrote that are not in the PDF",
        "",
        "A transcriber that generates text can also add text. Lines in the output whose",
        "words do not appear anywhere in pdfplumber's extraction.",
    ]),
    code("invented", [
        "pdf_words = set(re.findall(r'[a-z]{4,}', stage1.lower()))",
        "for name, t in runs.items():",
        "    extra = []",
        "    for line in t.splitlines():",
        "        words = re.findall(r'[a-z]{4,}', line.lower())",
        "        if len(words) >= 4 and sum(w not in pdf_words for w in words) >= len(words) // 2:",
        "            extra.append(line.strip()[:110])",
        "    print(f'{name}: {len(extra)} line(s) not grounded in the PDF')",
        "    for e in extra[:5]:",
        "        print('   ', repr(e))",
    ]),

    # ── 4 ─────────────────────────────────────────────────────────────────
    md("md-4", [
        "## 4. Conclusion",
        "",
        "LightOnOCR's native output is prose in Markdown, and its idea of formatting is",
        "Markdown's: headings and bold for anything that looks emphasised, `*…*` for the",
        "occasional italic, and no notion of underline at all — Markdown has none, so the",
        "model never learned one.",
        "",
        "- **Bold** is recovered as a *shape* (a `#` heading or `**`) rather than as a",
        "  measured property. Asking for `**` in the prompt shifts some headings to `**` but",
        "  does not make the count match the PDF.",
        "- **Italic** is caught only when short and inline. Whole italic paragraphs — the",
        "  ones that mark `section.description` in this document — come back plain in both",
        "  runs, regardless of prompt.",
        "- **Underline** is never produced, under either prompt. The model reports an",
        "  underlined phrase as bold or italic, or not at all.",
        "",
        "Unlike Docling, there is no lower level to recover from: the model's text is the",
        "only artefact, and the prompt is the only control. This is the nature of the",
        "approach, not a configuration problem — the extractor's docstring already records",
        "that the model ignores the `++` instruction, and this notebook measures by how much.",
    ]),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"built {NB}: {len(cells)} cells")
