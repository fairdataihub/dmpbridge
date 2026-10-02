"""One prompt version: its folder, its prompt on record, the model calls and the scores.

A version lives in <runs_dir>/<name>/ and holds, per model, the JSON it produced
(or the raw text if it never finished valid JSON), plus prompt.txt, run.json
(timings, tokens, why a model has no result), and after scoring scores.json,
details.csv and evaluation_results.xlsx. Nothing in a version is generated twice:
a saved result is loaded, a saved failure is reported, and a changed prompt under
an existing name is refused.
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from dmpbridge.models.ollama import OllamaModel
from dmpbridge.rda.score import metrics, reference_fields, score


def tag(model):
    """Model name as used in file names."""
    return model.replace(":", "-")


class Version:
    def __init__(self, name, sample, models, runs_dir="data/output/rda/runs",
                 host="http://localhost:11434", num_ctx=32768, max_tokens=8000):
        self.name, self.sample, self.models = name, sample, list(models)
        self.runs_dir, self.dir = Path(runs_dir), Path(runs_dir) / name
        self.host, self.num_ctx, self.max_tokens = host, num_ctx, max_tokens
        self.results = {}                              # model -> JSON (None when there is no result)
        info = self._read_json("run.json")
        self.timings = info.get("timings", {})
        self.not_run = {m: why for m, why in info.get("not_run", {}).items() if "no valid JSON" in why}
        self.system = self.make_prompt = self.schema = None
        saved = sorted(p.name for p in self.runs_dir.glob("*") if p.is_dir())
        print(f"version {name!r}: {'already saved - results will be loaded' if self.dir.exists() else 'new'}")
        print("saved versions:", ", ".join(saved) or "none yet")

    # ── Files ─────────────────────────────────────────────────────────────────
    def _read_json(self, name):
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _write_json(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def result_file(self, model):
        return self.dir / f"sample{self.sample}.rda.{tag(model)}.json"

    def raw_file(self, model):
        return self.dir / f"sample{self.sample}.rda.{tag(model)}.raw.txt"

    def _save_run_info(self):
        self._write_json("run.json", {
            "run": self.name, "date": datetime.now().isoformat(timespec="minutes"), "sample": self.sample,
            "models": self.models, "num_ctx": self.num_ctx, "timings": self.timings, "not_run": self.not_run})

    # ── The prompt ────────────────────────────────────────────────────────────
    def set_prompt(self, system, make_prompt, schema, schema_text):
        """Record the prompt. Refuses a changed prompt under an existing name."""
        self.system, self.make_prompt, self.schema = system, make_prompt, schema
        self.prompt_record = ("SYSTEM:\n" + system
                              + "\n\nPROMPT (the schema and the document text go in the braces):\n"
                              + make_prompt("{dmp_text}").replace(schema_text, "{schema_text}"))
        saved = self.dir / "prompt.txt"
        if saved.exists() and saved.read_text(encoding="utf-8") != self.prompt_record:
            names = ", ".join(sorted(p.name for p in self.runs_dir.glob("*") if p.is_dir()))
            raise SystemExit(f"The prompt is not the one saved for version {self.name!r}.\n"
                             f"Give this prompt a new name in the settings cell (saved versions: {names}),\n"
                             f"or delete {self.dir} if you really want to redo {self.name!r}.")
        self.dir.mkdir(parents=True, exist_ok=True)
        saved.write_text(self.prompt_record, encoding="utf-8")
        print(f"prompt saved -> {saved}")

    # ── Calling a model ───────────────────────────────────────────────────────
    def run(self, model, dmp_text, show=True):
        """One model on the prompt. The JSON (or None when there is no result)
        is kept in self.results[model] and printed when show is true."""
        result = self._run(model, dmp_text)
        self.results[model] = result
        self._save_run_info()
        if show and result is not None:
            print()
            print(json.dumps(result, indent=2, ensure_ascii=False))

    def _run(self, model, dmp_text):
        out = self.result_file(model)
        if out.exists():
            print(f"{model}: already saved for version {self.name!r}, loaded from {out}")
            t = self.timings.get(model)
            if t:
                print(f"(that run took {t['seconds']} s, {t['tokens_sent']:,} tokens sent, "
                      f"{t['tokens_generated']:,} generated)")
            return json.loads(out.read_text(encoding="utf-8"))
        if self.raw_file(model).exists():
            self.not_run.setdefault(model, "no valid JSON")
            print(f"{model}: already failed for version {self.name!r} - {self.not_run[model]}")
            print(f"(delete {self.raw_file(model).name} in the version's folder to try it again)")
            return None

        for other in self.models:                      # one model in VRAM at a time
            if other != model:
                subprocess.run(["ollama", "stop", other], check=False)

        # Load the model, then make sure all of it is on the GPU before the real call:
        # a model spilling onto the CPU turns seconds into hours.
        t0 = time.perf_counter()
        requests.post(f"{self.host}/api/generate", timeout=1800,
                      json={"model": model, "keep_alive": -1, "options": {"num_ctx": self.num_ctx}})
        load = time.perf_counter() - t0
        loaded = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
        placement = next((l for l in loaded.splitlines() if l.startswith(model)), "")
        if "100% GPU" not in placement:
            self.not_run[model] = f"not fully on the GPU (ollama ps says: {' '.join(placement.split()) or 'not loaded'})"
            print(f"{model}: SKIPPED - {self.not_run[model]}")
            subprocess.run(["ollama", "stop", model], check=False)
            return None

        llm = OllamaModel(model=model, host=self.host, num_ctx=self.num_ctx, num_predict=self.max_tokens)
        t0 = time.perf_counter()
        raw = llm.complete(self.system, self.make_prompt(dmp_text), schema=self.schema)
        elapsed = time.perf_counter() - t0
        s = llm.last_call
        self.timings[model] = {"seconds": round(elapsed), "model_load_seconds": round(load),
                               "tokens_sent": s["prompt_eval_count"], "tokens_generated": s["eval_count"]}
        print(f"{model}: {elapsed:.0f} s   (loading the model took {load:.0f} s)")
        print(f"tokens sent to the model: {s['prompt_eval_count']:,}   tokens generated: {s['eval_count']:,}")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            # The model did not finish a valid JSON - usually it looped and hit the token cap.
            cut = (f"it stopped at the {self.max_tokens:,}-token cap, so it never finished"
                   if s["done_reason"] == "length" else f"{e.msg} at character {e.pos:,}")
            self.not_run[model] = f"no valid JSON - {cut}"
            self.raw_file(model).write_text(raw, encoding="utf-8")
            print(f"{model}: FAILED - {self.not_run[model]}")
            print(f"what it wrote is kept in {self.raw_file(model)}")
            return None
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> {out}")
        return result

    # ── Scoring ───────────────────────────────────────────────────────────────
    def score(self, reference):
        """Score every result against the reference JSON. Saves scores.json,
        details.csv and evaluation_results.xlsx in the version's folder.
        -> (summary DataFrame, one row per model; details DataFrame, one row per field)."""
        import pandas as pd

        ref = json.loads(Path(reference).read_text(encoding="utf-8"))
        n_ref = len(reference_fields(ref))
        for m, why in self.not_run.items():
            print(f"not scored: {m} - {why}")
        rows, scores = [], {}
        for m, result in self.results.items():
            if result is None:
                continue
            judged = score(result, ref)
            rows += [{"model": m, **r} for r in judged]
            scores[m] = metrics(judged, n_ref)
        if not rows:
            raise SystemExit(f"No model produced a result for version {self.name!r}, so there is nothing to score.")
        details = pd.DataFrame(rows)
        summary = pd.DataFrame.from_dict(scores, orient="index").rename_axis("model")

        self._write_json("scores.json", {"run": self.name, "date": datetime.now().isoformat(timespec="minutes"),
                                         "sample": self.sample, "fields in reference": n_ref, "scores": scores})
        details.to_csv(self.dir / "details.csv", index=False)
        with pd.ExcelWriter(self.dir / "evaluation_results.xlsx") as xw:
            summary.to_excel(xw, sheet_name="summary")
            for m in scores:
                details[details["model"] == m].drop(columns="model").to_excel(xw, sheet_name=tag(m)[:31], index=False)
        print(f"version {self.name!r}, sample{self.sample}: the person filled in {n_ref} fields")
        print(f"saved -> {self.dir / 'scores.json'}, details.csv and evaluation_results.xlsx")
        return summary, details
