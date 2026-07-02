"""Configuration-driven experiment system.

An :class:`ExperimentConfig` is a complete, reproducible description of one
classification run: which strategy, model, prompt, and data to use.  Configs
live in ``experiments/*.yaml`` and can be loaded, inspected, and replayed
without touching any source code.

An :class:`Experiment` takes a config, builds the right strategy, runs it over
every sample PDF, saves the results, and optionally evaluates them.

Usage — Python API
------------------
    from dmpbridge.experiment import Experiment, ExperimentConfig

    exp    = Experiment.from_yaml("experiments/claude-opus-4-8-batch.yaml")
    paths  = exp.run()                 # classify + save JSON for each sample
    df, conf, errs = exp.evaluate()    # compare against manual labels

Usage — CLI
-----------
    dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml
    dmpbridge-experiment experiments/claude-opus-4-8-batch.yaml --evaluate
    dmpbridge-experiment --list
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# Default experiments directory (project root / experiments/)
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


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
        ``"batch"`` | ``"wholedoc"`` | ``"pdf_direct"``
    model:
        Model identifier passed to the provider.
    provider:
        ``"anthropic"`` | ``"ollama"`` | ``"openai"`` | ``"gemini"``
    host:
        Ollama base URL (ignored for cloud providers).
    prompt:
        Prompt variant to use.  ``"default"`` is the only supported value now;
        reserved for future A/B prompt experiments.
    batch_size:
        Number of blocks per LLM request.  Only applies to the batch strategy.
    context_size:
        Sliding-context window size for the batch strategy.
    pdf_dir:
        Directory that contains ``sample1.pdf`` … ``sample10.pdf``.
    out_dir:
        Directory where labeled JSON files are written.
    sample_start:
        First sample index to process (inclusive).
    sample_end:
        Last sample index to process (inclusive).
    """

    name:         str
    strategy:     str
    model:        str
    provider:     str

    host:         str = "http://localhost:11434"
    prompt:       str = "default"
    batch_size:   int = 10
    context_size: int = 3

    pdf_dir:      str = "data/pdfsamples"
    out_dir:      str = "data/llmlabeled"
    sample_start: int = 1
    sample_end:   int = 10

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def tag(self) -> str:
        """Output filename suffix that identifies this experiment.

        Matches the existing naming convention so notebooks load files without
        any changes::

            sample1_{tag}.json
        """
        _suffix = {
            "batch":      "batch",
            "wholedoc":   "whole_doc",
            "pdf_direct": "pdf",
        }
        suffix = _suffix.get(self.strategy, self.strategy)
        return f"{self.model.replace(':', '-')}_{suffix}"

    @property
    def sample_range(self) -> range:
        """Range of sample indices to process."""
        return range(self.sample_start, self.sample_end + 1)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Build a config from a plain dictionary."""
        return cls(**d)

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
            f"model={self.model!r}, provider={self.provider!r}, "
            f"tag={self.tag!r})"
        )


# ── Experiment ────────────────────────────────────────────────────────────────

class Experiment:
    """Run and evaluate one experiment defined by an :class:`ExperimentConfig`.

    The strategy is built lazily on first use so that importing this class does
    not trigger any network connections or API key checks.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config    = config
        self._strategy = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        """Load config from *path* and return a ready-to-run Experiment."""
        return cls(ExperimentConfig.from_yaml(path))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_strategy(self):
        if self._strategy is None:
            from .strategies import get_strategy
            cfg = self.config
            kwargs: dict = dict(provider=cfg.provider, model=cfg.model, host=cfg.host)
            if cfg.strategy == "batch":
                kwargs["batch_size"]   = cfg.batch_size
                kwargs["context_size"] = cfg.context_size
            self._strategy = get_strategy(cfg.strategy, **kwargs)
        return self._strategy

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> list[Path]:
        """Classify every sample PDF defined in the config and save results.

        Samples that already have an output file are skipped so the command is
        safe to re-run after a partial failure.

        Returns
        -------
        list[Path]
            Paths of the output JSON files that were written (or already existed).
        """
        strategy = self._get_strategy()
        cfg      = self.config
        out_dir  = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []

        for i in cfg.sample_range:
            label    = f"[sample{i}]"
            pdf_path = Path(cfg.pdf_dir) / f"sample{i}.pdf"
            out_path = out_dir / f"sample{i}_{cfg.tag}.json"

            if out_path.exists():
                logger.info("%s already exists — skipping", label)
                outputs.append(out_path)
                continue

            if not pdf_path.exists():
                logger.warning("%s PDF not found: %s", label, pdf_path)
                continue

            logger.info("%s running experiment %r …", label, cfg.name)
            blocks = strategy.run(pdf_path)
            out_path.write_text(
                json.dumps(blocks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("%s %d blocks → %s", label, len(blocks), out_path.name)
            outputs.append(out_path)

        logger.info("Done — %d/%d samples processed.", len(outputs), len(cfg.sample_range))
        return outputs

    def evaluate(self):
        """Evaluate results against manual labels using :func:`~dmpbridge.evaluate.load_method`.

        Returns
        -------
        tuple[pd.DataFrame, dict, pd.DataFrame] | tuple[None, None, None]
            ``(accuracy_df, confusion_dict, errors_df)``
        """
        from .evaluate import load_method
        return load_method(self.config.tag)

    def run_and_evaluate(self):
        """Run the experiment then evaluate.  Convenience wrapper."""
        self.run()
        return self.evaluate()

    def summary(self) -> None:
        """Print a one-line accuracy summary to stdout."""
        df, conf, _ = self.evaluate()
        if df is None:
            print(f"[{self.config.name}]  no results found — run first")
            return
        tc = int(df["correct"].sum())
        tn = int(df["total"].sum())
        print(f"[{self.config.name}]  {tc}/{tn}  ({tc/tn*100:.1f}%)  tag={self.config.tag!r}")

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
            cfg = ExperimentConfig.from_yaml(p)
            print(f"  {p.name:<45}  {cfg.strategy:<12}  {cfg.model}")
        sys.exit(0)

    if not args.config:
        ap.print_help()
        sys.exit(1)

    exp = Experiment.from_yaml(args.config)
    print(f"\nExperiment : {exp.config.name}")
    print(f"Strategy   : {exp.config.strategy}")
    print(f"Model      : {exp.config.model}  ({exp.config.provider})")
    print(f"Tag        : {exp.config.tag}")
    print(f"Samples    : {exp.config.sample_start}–{exp.config.sample_end}\n")

    if not args.evaluate_only:
        exp.run()

    if args.evaluate or args.evaluate_only:
        exp.summary()


if __name__ == "__main__":
    main()
