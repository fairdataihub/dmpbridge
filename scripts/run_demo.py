"""Run the pipeline using demo/config.yaml, then save each stage's result
into its own subfolder under demo/output/ — labeled/, structured/, final/ —
mirroring the same stage names data/output/ itself uses.

Edit demo/config.yaml first (model, extractor, sample range), then:

    python scripts/run_demo.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpbridge.core import paths as P
from dmpbridge.evaluation.experiment import ExperimentConfig, Experiment

CONFIG_PATH = Path(__file__).parent.parent / "demo" / "config.yaml"
OUTPUT_DIR  = Path(__file__).parent.parent / "demo" / "output"

# (subfolder name, function that resolves that stage's real file for one
# (tag, sample) pair) — one entry per stage copied into demo/output/.
STAGES = [
    ("labeled",    P.labeled_path),
    ("structured", P.structured_path),
    ("final",      P.final_path),
]


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"No config at {CONFIG_PATH} — see demo/config.yaml")

    cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
    print(f"Demo      : {cfg.name}")
    print(f"Model     : {', '.join(cfg.models)}")
    print(f"Extractor : {', '.join(cfg.extractors)}")
    print(f"Samples   : {cfg.sample_start}-{cfg.sample_end}\n")

    exp = Experiment(cfg)
    exp.run()

    counts = {name: 0 for name, _ in STAGES}
    for model in cfg.models:
        for extractor in cfg.extractors:
            tag = cfg.tag_for(model, extractor)
            for n in cfg.sample_range:
                for stage_name, resolve in STAGES:
                    src = resolve(tag, n)
                    if not src.exists():
                        continue
                    dest_dir = OUTPUT_DIR / stage_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest_dir / f"sample{n}.json")
                    counts[stage_name] += 1

    print(f"Saved to {OUTPUT_DIR}/:")
    for stage_name, _ in STAGES:
        print(f"  {stage_name}/  {counts[stage_name]} file(s)")


if __name__ == "__main__":
    main()
