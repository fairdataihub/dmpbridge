"""Pipeline stages 5 and 6 for one sample - shared by every runner.

``dmpbridge-pipeline``, ``dmpbridge-rda`` and ``scripts/run_demo.py`` (with an
``rda:`` block in demo/config.yaml) all call :func:`run_rda_stages`, so they
write the same files:

    5_rda/sampleN.json          the RDA JSON
    5_rda/sampleN.raw.txt       the answer, when it is not valid JSON
    6_merged/sampleN.json       {"dmp": <stage 5>, "narrative": <stage 4>}

The folders are not keyed by model, so beside each ``sampleN.json`` a
``sampleN.run.json`` records the models, extractor and prompt that made it. Stage 5
reuses a saved file only when that record matches the current run; a run with a
different RDA model, extractor or prompt replaces it. Stage 6 is rewritten every
time, since it is only a merge.
"""
import json
import time
from pathlib import Path
from typing import Optional

from ..core import config, paths
from ..utils import get_logger
from .extract import PROMPT_VERSION, RdaExtractionError, extract_rda, merge

logger = get_logger(__name__)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_run_info(stage_file: Path) -> Optional[dict]:
    """The ``sampleN.run.json`` record beside a stage 5 or 6 file, if there is one."""
    info = paths.run_info_path(stage_file)
    return json.loads(info.read_text(encoding="utf-8")) if info.exists() else None


def run_rda_stages(strategy, pdf_path: Path, sample: int, rda_model: str,
                   narrative_model: str, extractor: str,
                   host: str = config.HOST) -> tuple[Optional[Path], Optional[Path]]:
    """Run stage 5 (RDA JSON) and stage 6 (merge) for one sample.

    *strategy* is a WholeDocStrategy; only its :meth:`extract_text` is used, so
    the RDA model reads exactly the text the narrative model read. Returns the
    stage-5 and stage-6 paths, each ``None`` when that stage could not be written.
    """
    label = f"[sample{sample}]"
    rda_out = paths.rda_path(sample)
    this_run = {"rda_model": rda_model, "extractor": extractor, "prompt": PROMPT_VERSION}
    saved = read_run_info(rda_out) if rda_out.exists() else None

    if saved == this_run:
        logger.info("%s RDA JSON already exists for %s - skipping extraction", label, rda_model)
        rda = json.loads(rda_out.read_text(encoding="utf-8"))
    elif not pdf_path.exists():
        logger.warning("%s PDF not found: %s", label, pdf_path)
        return None, None
    else:
        if rda_out.exists():
            changed = ", ".join(f"{k} {saved[k]} -> {v}" for k, v in this_run.items()
                                if saved.get(k) != v) if saved else "no run record"
            logger.info("%s saved RDA JSON is from a different run (%s) - replacing it", label, changed)
            rda_out.unlink()
            paths.run_info_path(rda_out).unlink(missing_ok=True)
        text = strategy.extract_text(pdf_path)
        if not text:
            return None, None
        t0 = time.perf_counter()
        try:
            rda, info = extract_rda(text, rda_model, host=host)
        except RdaExtractionError as e:
            raw_out = rda_out.with_name(f"sample{sample}.raw.txt")
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            raw_out.write_text(e.raw, encoding="utf-8")
            logger.error("%s %s - answer saved to %s", label, e, raw_out)
            return None, None
        _write_json(rda_out, rda)
        _write_json(paths.run_info_path(rda_out), this_run)
        rda_out.with_name(f"sample{sample}.raw.txt").unlink(missing_ok=True)   # an earlier failure
        logger.info("%s RDA JSON in %.1f s (%s output tokens) -> %s",
                    label, time.perf_counter() - t0, info.get("eval_count"), rda_out)

    narrative_in = paths.final_path(paths.make_tag(narrative_model, extractor), sample)
    if not narrative_in.exists():
        logger.warning("%s no narrative JSON at %s - not merged", label, narrative_in)
        return rda_out, None
    narrative = json.loads(narrative_in.read_text(encoding="utf-8"))
    merged_out = paths.merged_path(sample)
    _write_json(merged_out, merge(narrative, rda))
    _write_json(paths.run_info_path(merged_out), {"narrative_model": narrative_model, **this_run})
    logger.info("%s merged -> %s", label, merged_out)
    return rda_out, merged_out
