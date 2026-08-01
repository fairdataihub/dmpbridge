"""Configuration-driven experiment system.

An :class:`ExperimentConfig` is a complete, reproducible description of one
classification run: which strategy, model, prompt, and data to use.  Configs
live in ``experiments/*.yaml`` and can be loaded, inspected, and replayed
without touching any source code.

An :class:`Experiment` takes a config, builds the right strategy, runs it over
every sample PDF, saves the results, and optionally evaluates them.

Usage — Python API
------------------
    from dmpbridge.evaluation.experiment import Experiment, ExperimentConfig

    exp    = Experiment.from_yaml("experiments/llama3.3-70b-wholedoc.yaml")
    paths  = exp.run()                 # classify + save JSON for each sample
    df, conf, errs = exp.evaluate()    # compare against manual labels

Usage — CLI
-----------
    dmpbridge-experiment experiments/llama3.3-70b-wholedoc.yaml
    dmpbridge-experiment experiments/llama3.3-70b-wholedoc.yaml --evaluate
    dmpbridge-experiment --list
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..core.pipeline import run_and_save
from ..utils import get_logger, setup_logging

logger = get_logger(__name__)

# Default experiments directory (project root / experiments/)
EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Complete, reproducible description of one classification experiment.

    All fields that affect the output are captured so that replaying the same
    config on the same data always produces the same result.

    Attributes
    ----------
    name:
        Human-readable experiment name shown in reports.
    strategy:
        ``"wholedoc"`` — the only supported strategy.
    model:
        Model identifier passed to the provider.
    provider:
        ``"ollama"`` — only supported provider.
    host:
        Ollama base URL (ignored for cloud providers).
    prompt:
        Prompt variant to use.  ``"default"`` is the only supported value now;
        reserved for future A/B prompt experiments.
    pdf_dir:
        Directory that contains ``sample1.pdf`` … ``sample10.pdf``.
    out_dir:
        Directory where labeled JSON files are written.
    sample_start:
        First sample index to process (inclusive).
    sample_end:
        Last sample index to process (inclusive).
    """

    name:       str
    strategy:   str
    provider:   str

    models:     list = field(default_factory=list)
    extractors: list = field(default_factory=lambda: ["pdfplumber"])

    host:         str = "http://localhost:11434"
    prompt:       str = "default"
    pdf_dir:      str = "data/input/pdfs"
    out_dir:      str = "data/output/labeled"
    sample_start: int = 1
    sample_end:   int = 10

    # ── Derived properties ────────────────────────────────────────────────────

    def tag_for(self, model: str, extractor: str) -> str:
        """Output directory suffix for one (model, extractor) combination.

        The extractor name is always included, matching the folder layout
        used everywhere else in the pipeline (e.g. ``dmpbridge-wholedoc``)::

            llama3.1-8b_pdfplumber_whole_doc
            llama3.1-8b_docling_whole_doc
            llama3.1-8b_lighton_whole_doc
        """
        model_slug = model.replace(":", "-")
        suffix     = "whole_doc" if self.strategy == "wholedoc" else self.strategy
        return f"{model_slug}_{extractor}_{suffix}"

    @property
    def tags(self) -> list[str]:
        """Tags for every (model, extractor) combination."""
        return [self.tag_for(m, e) for m in self.models for e in self.extractors]

    @property
    def sample_range(self) -> range:
        """Range of sample indices to process."""
        return range(self.sample_start, self.sample_end + 1)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file.

        Backward-compat promotions applied before construction:
        - ``model: x``     → ``models: [x]``
        - ``extractor: x`` → ``extractors: [x]``
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**cls._normalise(data))

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Build a config from a plain dictionary."""
        return cls(**cls._normalise(dict(d)))

    @staticmethod
    def _normalise(data: dict) -> dict:
        """Promote/strip fields for backward compat."""
        if "model" in data and "models" not in data:
            data["models"] = [data.pop("model")]
        elif "model" in data:
            data.pop("model")
        if "extractor" in data and "extractors" not in data:
            data["extractors"] = [data.pop("extractor")]
        elif "extractor" in data:
            data.pop("extractor")
        # Removed fields — drop silently so old YAMLs still load.
        data.pop("few_shot_samples", None)
        data.pop("eval_samples", None)
        return data

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file."""
        Path(path).write_text(
            yaml.dump(asdict(self), default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def to_dict(self) -> dict:
        """Return this config as a plain dictionary (excludes derived properties)."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"ExperimentConfig("
            f"name={self.name!r}, strategy={self.strategy!r}, "
            f"models={self.models!r}, extractors={self.extractors!r}, "
            f"provider={self.provider!r})"
        )


# ── Experiment ────────────────────────────────────────────────────────────────

class Experiment:
    """Run and evaluate one experiment defined by an :class:`ExperimentConfig`.

    Strategies are built lazily per extractor so that importing this class does
    not trigger any network connections or API key checks.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config      = config
        self._strategies: dict = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        """Load config from *path* and return a ready-to-run Experiment."""
        return cls(ExperimentConfig.from_yaml(path))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_strategy(self, model: str, extractor: str):
        key = (model, extractor)
        if key not in self._strategies:
            from ..strategies import get_strategy
            cfg = self.config
            self._strategies[key] = get_strategy(
                cfg.strategy,
                provider=cfg.provider,
                model=model,
                host=cfg.host,
                extractor=extractor,
            )
        return self._strategies[key]

    def _run_combination(self, model: str, extractor: str) -> list[Path]:
        """Run all samples for one (model, extractor) pair."""
        strategy = self._get_strategy(model, extractor)
        cfg      = self.config
        out_dir  = Path(cfg.out_dir) / cfg.tag_for(model, extractor)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []

        for i in cfg.sample_range:
            label       = f"[sample{i}]"
            pdf_path    = Path(cfg.pdf_dir) / f"sample{i}.pdf"
            out_path    = out_dir / f"sample{i}.json"
            struct_path = out_dir / f"sample{i}_structured.json"

            if out_path.exists():
                logger.info("%s already exists — skipping", label)
                outputs.append(out_path)
                continue

            logger.info("%s running [%s / %s] …", label, model, extractor)
            blocks = run_and_save(strategy, pdf_path, out_path, struct_path)

            if blocks is None:
                logger.warning("%s PDF not found: %s", label, pdf_path)
                continue

            logger.info("%s %d blocks → %s", label, len(blocks), out_path.name)
            logger.info("%s structured JSON → %s", label, struct_path.name)
            outputs.append(out_path)

        logger.info(
            "Done [%s / %s] — %d/%d samples processed.",
            model, extractor, len(outputs), len(cfg.sample_range),
        )
        return outputs

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict[str, dict[str, list[Path]]]:
        """Classify every sample PDF for every (model, extractor) combination.

        Samples that already have an output file are skipped, so the command
        is safe to re-run after a partial failure.

        Returns
        -------
        dict[str, dict[str, list[Path]]]
            ``{model: {extractor: [output_paths]}}``
        """
        cfg = self.config
        return {
            m: {e: self._run_combination(m, e) for e in cfg.extractors}
            for m in cfg.models
        }

    def evaluate(self) -> dict[str, dict[str, tuple]]:
        """Evaluate results for every (model, extractor) pair against manual labels.

        Returns
        -------
        dict[str, dict[str, tuple]]
            ``{model: {extractor: (accuracy_df, confusion_dict, errors_df)}}``
        """
        from .evaluate import load_method
        cfg = self.config
        return {
            m: {e: load_method(cfg.tag_for(m, e)) for e in cfg.extractors}
            for m in cfg.models
        }

    def run_and_evaluate(self) -> dict[str, dict[str, tuple]]:
        """Run the experiment then evaluate.  Convenience wrapper."""
        self.run()
        return self.evaluate()

    def summary(self) -> None:
        """Print one accuracy line per (model, extractor) combination."""
        for model, ext_results in self.evaluate().items():
            for extractor, (df, _, _) in ext_results.items():
                tag = self.config.tag_for(model, extractor)
                if df is None:
                    print(f"  [{model} / {extractor}]  no results — run first")
                    continue
                tc = int(df["correct"].sum())
                tn = int(df["total"].sum())
                print(f"  [{model} / {extractor}]  {tc}/{tn}  ({tc/tn*100:.1f}%)  tag={tag!r}")

    def __repr__(self) -> str:
        return f"Experiment({self.config})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_experiments(experiments_dir: Path = EXPERIMENTS_DIR) -> list[Path]:
    """Return all YAML files in *experiments_dir*, sorted by name."""
    if not experiments_dir.exists():
        return []
    return sorted(experiments_dir.glob("*.yaml"))


def load_all(experiments_dir: Path = EXPERIMENTS_DIR) -> list[Experiment]:
    """Load every experiment config in *experiments_dir*."""
    return [Experiment.from_yaml(p) for p in list_experiments(experiments_dir)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: dmpbridge-experiment."""
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Run or evaluate a DMPBridge experiment defined by a YAML config."
    )
    ap.add_argument(
        "config",
        nargs="?",
        help="Path to an experiment YAML file.",
    )
    ap.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Evaluate results against manual labels after running.",
    )
    ap.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip running — only print accuracy summary for existing results.",
    )
    ap.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiment configs and exit.",
    )
    args = ap.parse_args()

    if args.list:
        paths = list_experiments()
        if not paths:
            print(f"No experiments found in {EXPERIMENTS_DIR}")
            sys.exit(0)
        print(f"Experiments in {EXPERIMENTS_DIR}:\n")
        for p in paths:
            cfg    = ExperimentConfig.from_yaml(p)
            models = ", ".join(cfg.models)
            exts   = ", ".join(cfg.extractors)
            print(f"  {p.name:<45}  {cfg.strategy:<10}  models=[{models}]  extractors=[{exts}]")
        sys.exit(0)

    if not args.config:
        ap.print_help()
        sys.exit(1)

    exp = Experiment.from_yaml(args.config)
    cfg = exp.config
    n_combos = len(cfg.models) * len(cfg.extractors)
    print(f"\nExperiment : {cfg.name}")
    print(f"Strategy   : {cfg.strategy}")
    print(f"Models     : {', '.join(cfg.models)}")
    print(f"Extractors : {', '.join(cfg.extractors)}")
    print(f"Provider   : {cfg.provider}")
    print(f"Matrix     : {n_combos} combinations ({len(cfg.models)} models × {len(cfg.extractors)} extractors)")
    print(f"Samples    : {cfg.sample_start}–{cfg.sample_end}\n")

    if not args.evaluate_only:
        exp.run()

    if args.evaluate or args.evaluate_only:
        exp.summary()


if __name__ == "__main__":
    main()
