"""Run the pipeline using demo/config.yaml, then save each stage's result
into its own subfolder under demo/output/ — extracted/, labeled/, structured/,
final/, and with an ``rda:`` block in the config also rda/ and merged/ —
mirroring the six stage folders data/output/ itself uses.

The ``narrative:`` block picks the model for stages 1-4, which run through the
experiment runner unchanged; the ``rda:`` block picks the model for stages 5-6,
which run after it.

Edit demo/config.yaml first (model, extractor, sample range), then:

    python scripts/run_demo.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpbridge.cli.rda_evaluate_cmd import evaluate_tag
from dmpbridge.core import paths as P
from dmpbridge.core.pipeline import normalize_fallbacks
from dmpbridge.evaluation.experiment import Experiment
from dmpbridge.rda.config import load_pipeline_config
from dmpbridge.rda.stages import run_rda_stages
from dmpbridge.strategies.wholedoc import WholeDocStrategy

CONFIG_PATH = Path(__file__).parent.parent / "demo" / "config.yaml"
OUTPUT_DIR  = Path(__file__).parent.parent / "demo" / "output"


def stage_files(cfg, rda_model, model: str, extractor: str, n: int) -> list[tuple[str, Path]]:
    """(subfolder name, that stage's real file) for one model, extractor and sample."""
    tag = cfg.tag_for(model, extractor)
    files = [
        ("extracted",  P.extracted_path(extractor, n)),
        ("labeled",    P.labeled_path(tag, n)),
        ("structured", P.structured_path(tag, n)),
        ("final",      P.final_path(tag, n)),
    ]
    if rda_model:
        files += [
            ("rda",    P.rda_path(P.rda_tag(rda_model, extractor), n)),
            ("merged", P.merged_path(P.merged_tag(model, rda_model, extractor), n)),
        ]
    return files


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No config at {CONFIG_PATH} — see demo/config.yaml")

    cfg, rda_model = load_pipeline_config(CONFIG_PATH)

    print(f"Demo            : {cfg.name}")
    print(f"Narrative model : {', '.join(cfg.models)}")
    print(f"RDA model       : {rda_model or '(none - no rda: block)'}")
    print(f"Extractor       : {', '.join(cfg.extractors)}")
    print(f"Samples         : {cfg.sample_start}-{cfg.sample_end}\n")

    # Stages 1-4: the narrative pipeline.
    Experiment(cfg).run()

    # Stages 5-6: the RDA JSON from the same text, merged with each model's stage 4.
    if rda_model:
        for extractor in cfg.extractors:
            strategy = WholeDocStrategy(
                model=rda_model, host=cfg.host, extractor=extractor,
                cache_dir=P.EXTRACTED_DIR / extractor,
                fallback_extractors=normalize_fallbacks(cfg.fallback, extractor),
            )
            for model in cfg.models:
                for n in cfg.sample_range:
                    run_rda_stages(strategy, Path(cfg.pdf_dir) / f"sample{n}.pdf", n,
                                   rda_model, model, extractor, host=cfg.host)
            # Score the RDA JSON for the samples that have a hand-made reference.
            evaluate_tag(P.rda_tag(rda_model, extractor), list(cfg.sample_range))

    counts: dict[str, int] = {}
    for model in cfg.models:
        for extractor in cfg.extractors:
            for n in cfg.sample_range:
                for stage_name, src in stage_files(cfg, rda_model, model, extractor, n):
                    counts.setdefault(stage_name, 0)
                    if not src.exists():
                        continue
                    dest_dir = OUTPUT_DIR / stage_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest_dir / f"sample{n}.json")
                    counts[stage_name] += 1

    print(f"Saved to {OUTPUT_DIR}/:")
    for stage_name, count in counts.items():
        print(f"  {stage_name}/  {count} file(s)")


if __name__ == "__main__":
    main()
