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
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

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

    name:         str
    strategy:     str
    model:        str
    provider:     str

    host:         str  = "http://localhost:11434"
    prompt:       str  = "default"
    extractors:   list = field(default_factory=lambda: ["pdfplumber"])

    pdf_dir:      str = "data/input/pdfs"
    out_dir:      str = "data/output/labeled"
    sample_start: int = 1
    sample_end:   int = 10

    # Leave-2-out rotation fields — empty list means "use all samples / use hardcoded examples".
    # few_shot_samples: gold sample IDs to extract dynamic few-shot examples from.
    # eval_samples:     explicit eval set; when empty, all samples except few_shot_samples are used.
    few_shot_samples: list = field(default_factory=list)
    eval_samples:     list = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    def tag_for(self, extractor: str) -> str:
        """Output directory suffix for one extractor within this experiment.

        pdfplumber keeps the original format for backward compatibility with
        existing output data and notebooks::

            llama3.1-8b_whole_doc           ← pdfplumber (unchanged)
            llama3.1-8b_docling_whole_doc   ← docling
            llama3.1-8b_lighton_whole_doc   ← lighton
        """
        model_slug = self.model.replace(":", "-")
        suffix     = "whole_doc" if self.strategy == "wholedoc" else self.strategy
        if extractor == "pdfplumber":
            return f"{model_slug}_{suffix}"
        return f"{model_slug}_{extractor}_{suffix}"

    @property
    def tags(self) -> list[str]:
        """Tags for all configured extractors."""
        return [self.tag_for(e) for e in self.extractors]

    @property
    def sample_range(self) -> range:
        """Range of sample indices to process."""
        return range(self.sample_start, self.sample_end + 1)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file.

        Handles backward-compat: an old ``extractor: pdfplumber`` scalar field
        is silently promoted to ``extractors: [pdfplumber]``.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if "extractor" in data and "extractors" not in data:
            data["extractors"] = [data.pop("extractor")]
        elif "extractor" in data:
            data.pop("extractor")
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Build a config from a plain dictionary."""
        d = dict(d)
        if "extractor" in d and "extractors" not in d:
            d["extractors"] = [d.pop("extractor")]
        elif "extractor" in d:
            d.pop("extractor")
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
            f"extractors={self.extractors!r}, model={self.model!r}, "
            f"provider={self.provider!r}, tags={self.tags!r})"
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

    def _get_strategy(self, extractor: str):
        if extractor not in self._strategies:
            from ..strategies import get_strategy
            cfg    = self.config
            kwargs: dict = {
                "provider":  cfg.provider,
                "model":     cfg.model,
                "host":      cfg.host,
                "extractor": extractor,
            }
            if cfg.few_shot_samples:
                from ..prompts.few_shot import build_few_shot_examples
                from ..prompts.system import build_system_prompt
                examples = build_few_shot_examples(cfg.few_shot_samples)
                kwargs["system_prompt"] = build_system_prompt(examples)
                logger.info(
                    "Dynamic few-shot prompt built from samples %s", cfg.few_shot_samples
                )
            self._strategies[extractor] = get_strategy(cfg.strategy, **kwargs)
        return self._strategies[extractor]

    def _run_extractor(self, extractor: str) -> list[Path]:
        """Run all samples for one extractor and return output paths."""
        strategy = self._get_strategy(extractor)
        cfg      = self.config
        out_dir  = Path(cfg.out_dir) / cfg.tag_for(extractor)
        out_dir.mkdir(parents=True, exist_ok=True)

        few_shot_set = set(cfg.few_shot_samples)
        eval_set     = set(cfg.eval_samples) if cfg.eval_samples else set(cfg.sample_range) - few_shot_set

        outputs: list[Path] = []

        for i in cfg.sample_range:
            if i not in eval_set:
                logger.info("[sample%d] reserved for few-shot examples — skipping", i)
                continue

            label    = f"[sample{i}]"
            pdf_path = Path(cfg.pdf_dir) / f"sample{i}.pdf"
            out_path = out_dir / f"sample{i}.json"

            if out_path.exists():
                logger.info("%s already exists — skipping", label)
                outputs.append(out_path)
                continue

            if not pdf_path.exists():
                logger.warning("%s PDF not found: %s", label, pdf_path)
                continue

            logger.info("%s running experiment %r [%s] …", label, cfg.name, extractor)
            blocks = strategy.run(pdf_path)
            out_path.write_text(
                json.dumps(blocks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("%s %d blocks → %s", label, len(blocks), out_path.name)

            from ..core.converter import to_structured
            struct_path = out_dir / f"sample{i}_structured.json"
            struct_path.write_text(
                json.dumps(to_structured(blocks), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("%s structured JSON → %s", label, struct_path.name)
            outputs.append(out_path)

        logger.info(
            "Done [%s] — %d/%d samples processed.",
            extractor, len(outputs), len(cfg.sample_range),
        )
        return outputs

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict[str, list[Path]]:
        """Classify every sample PDF for every configured extractor.

        Samples that already have an output file are skipped so the command is
        safe to re-run after a partial failure.

        Returns
        -------
        dict[str, list[Path]]
            ``{extractor_name: [output_paths]}`` for each extractor.
        """
        return {e: self._run_extractor(e) for e in self.config.extractors}

    def evaluate(self) -> dict[str, tuple]:
        """Evaluate results for every extractor against manual labels.

        Returns
        -------
        dict[str, tuple[DataFrame, dict, DataFrame] | tuple[None, None, None]]
            ``{extractor_name: (accuracy_df, confusion_dict, errors_df)}``
        """
        from .evaluate import load_method
        return {e: load_method(self.config.tag_for(e)) for e in self.config.extractors}

    def run_and_evaluate(self) -> dict[str, tuple]:
        """Run the experiment then evaluate.  Convenience wrapper."""
        self.run()
        return self.evaluate()

    def summary(self) -> None:
        """Print one accuracy line per extractor to stdout."""
        for extractor, (df, _, _) in self.evaluate().items():
            tag = self.config.tag_for(extractor)
            if df is None:
                print(f"[{self.config.name} / {extractor}]  no results — run first")
                continue
            tc = int(df["correct"].sum())
            tn = int(df["total"].sum())
            print(
                f"[{self.config.name} / {extractor}]  "
                f"{tc}/{tn}  ({tc/tn*100:.1f}%)  tag={tag!r}"
            )

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
            exts = ", ".join(cfg.extractors)
            print(f"  {p.name:<45}  {cfg.strategy:<12}  {cfg.model:<22}  [{exts}]")
        sys.exit(0)

    if not args.config:
        ap.print_help()
        sys.exit(1)

    exp = Experiment.from_yaml(args.config)
    cfg = exp.config
    print(f"\nExperiment : {cfg.name}")
    print(f"Strategy   : {cfg.strategy}")
    print(f"Extractors : {', '.join(cfg.extractors)}")
    print(f"Model      : {cfg.model}  ({cfg.provider})")
    print(f"Tags       : {', '.join(cfg.tags)}")
    print(f"Samples    : {cfg.sample_start}–{cfg.sample_end}\n")

    if not args.evaluate_only:
        exp.run()

    if args.evaluate or args.evaluate_only:
        exp.summary()


if __name__ == "__main__":
    main()
