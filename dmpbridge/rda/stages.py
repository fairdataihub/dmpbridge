"""Pipeline stages 5 and 6 for one sample - shared by every runner.

``dmpbridge-pipeline``, ``dmpbridge-rda`` and ``scripts/run_demo.py`` (with an
``rda:`` block in demo/config.yaml) all call :func:`run_rda_stages`, so they
write the same files:

    5_rda/<rda-model>_<extractor>_rda/sampleN.json      the RDA JSON
    5_rda/<rda-model>_<extractor>_rda/sampleN.raw.txt   the answer, when not valid JSON
    6_merged/<narrative-model>+<rda-model>_<extractor>/sampleN.json

Stage 5 is cached by file existence like stages 2-4; stage 6 is rewritten every
time, since it is only a merge.
"""
import json
import time
from pathlib import Path
from typing import Optional

from ..core import config, paths
from ..utils import get_logger
from .extract import RdaExtractionError, extract_rda, merge

logger = get_logger(__name__)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_rda_stages(strategy, pdf_path: Path, sample: int, rda_model: str,
                   narrative_model: str, extractor: str,
                   host: str = config.HOST) -> tuple[Optional[Path], Optional[Path]]:
    """Run stage 5 (RDA JSON) and stage 6 (merge) for one sample.

    *strategy* is a WholeDocStrategy; only its :meth:`extract_text` is used, so
    the RDA model reads exactly the text the narrative model read. Returns the
    stage-5 and stage-6 paths, each ``None`` when that stage could not be written.
    """
    label = f"[sample{sample}]"
    rda_out = paths.rda_path(paths.rda_tag(rda_model, extractor), sample)

    if rda_out.exists():
        logger.info("%s RDA JSON already exists - skipping extraction", label)
        rda = json.loads(rda_out.read_text(encoding="utf-8"))
    elif not pdf_path.exists():
        logger.warning("%s PDF not found: %s", label, pdf_path)
        return None, None
    else:
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
        logger.info("%s RDA JSON in %.1f s (%s output tokens) -> %s",
                    label, time.perf_counter() - t0, info.get("eval_count"), rda_out)

    narrative_in = paths.final_path(paths.make_tag(narrative_model, extractor), sample)
    if not narrative_in.exists():
        logger.warning("%s no narrative JSON at %s - not merged", label, narrative_in)
        return rda_out, None
    narrative = json.loads(narrative_in.read_text(encoding="utf-8"))
    merged_out = paths.merged_path(paths.merged_tag(narrative_model, rda_model, extractor), sample)
    _write_json(merged_out, merge(narrative, rda))
    logger.info("%s merged -> %s", label, merged_out)
    return rda_out, merged_out
