#!/usr/bin/env python3
"""Single-experiment CLI wrapper for Phase 7 runners.

Dispatches to the right :class:`ExperimentRunner` subclass based on the
YAML's ``experiment_name`` field. Called from
``scripts/repro_paper.sh`` (one invocation per RQ) and from
``python -m experiments`` (see :mod:`experiments.__main__`).

Usage::

    python -m experiments \\
        --config experiments/configs/rq1_headline.yaml \\
        --output results/raw/rq1_headline.csv \\
        --manifest results/manifests/rq1_headline.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Self-bootstrap so ``python scripts/run_experiment.py`` works without
# needing the package to be pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from experiments.runners._base import ExperimentRunner

__all__ = ["main"]


# experiment_name (from the YAML) → import path of the Runner class.
_RUNNER_REGISTRY: dict[str, tuple[str, str]] = {
    "rq1_headline": ("experiments.runners.rq1_headline", "RQ1HeadlineRunner"),
    "rq2_kr_pareto": (
        "experiments.runners.rq2_kr_pareto",
        "RQ2KRParetoRunner",
    ),
    "rq3_close_analogues": (
        "experiments.runners.rq3_close_analogues",
        "RQ3CloseAnaloguesRunner",
    ),
    "rq4_ilp_gap": ("experiments.runners.rq4_ilp_gap", "RQ4ILPGapRunner"),
    "rq5_persistent": (
        "experiments.runners.rq5_persistent",
        "RQ5PersistentRunner",
    ),
    "ablation_ordering": (
        "experiments.runners.ablation_ordering",
        "AblationOrderingRunner",
    ),
    "ablation_solver": (
        "experiments.runners.ablation_solver",
        "AblationSolverRunner",
    ),
    "ablation_budget": (
        "experiments.runners.ablation_budget",
        "AblationBudgetRunner",
    ),
}


def _load_runner_class(experiment_name: str) -> type[ExperimentRunner]:
    if experiment_name not in _RUNNER_REGISTRY:
        known = ", ".join(sorted(_RUNNER_REGISTRY))
        raise ValueError(
            f"unknown experiment_name {experiment_name!r}; " f"expected one of: {known}"
        )
    module_path, class_name = _RUNNER_REGISTRY[experiment_name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single Phase 7 experiment from its YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML (e.g. experiments/configs/rq1_headline.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path. Appended to; safe to resume across runs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL manifest path. Tracks completed cells for resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help=(
            "Skip cells already recorded in the manifest. Default true; "
            "pass --no-resume to ignore the manifest and re-run everything."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore the manifest and re-run every cell from scratch.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on the number of cells to execute this run.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Override per-rollout parallelism. Defaults to the YAML's "
            "rollout_parallel_jobs value."
        ),
    )
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=None,
        help="Override the MovingAI benchmark root (default: <repo>/benchmarks/).",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        print(
            f"error: {args.config} did not parse to a mapping",
            file=sys.stderr,
        )
        return 2
    experiment_name = str(config.get("experiment_name", "")).strip()
    if not experiment_name:
        print(
            f"error: {args.config} has no experiment_name field",
            file=sys.stderr,
        )
        return 2

    runner_cls = _load_runner_class(experiment_name)
    runner = runner_cls(
        config_path=args.config,
        output_csv=args.output,
        manifest_path=args.manifest,
        benchmarks_root=args.benchmarks_root,
    )

    # --no-resume: erase the manifest so resume_from_manifest() returns
    # an empty completed set. The CSV is left intact (append semantics).
    if not args.resume and args.manifest.exists():
        args.manifest.unlink()

    # Honour --max-cells by trimming enumerate_cells via a small
    # adapter. This keeps the runner classes themselves agnostic of
    # the cap.
    if args.max_cells is not None:
        cap = int(args.max_cells)
        original_enumerate = runner.enumerate_cells

        def _capped_enumerate() -> object:
            for i, cell in enumerate(original_enumerate()):
                if i >= cap:
                    break
                yield cell

        runner.enumerate_cells = _capped_enumerate  # type: ignore[method-assign]

    # --jobs surface-area: not every runner reads it from cfg, so we
    # poke it into the loaded YAML's ``rollout_parallel_jobs`` field
    # for the duration of the run. Best-effort only.
    if args.jobs is not None:
        runner.load_config()["rollout_parallel_jobs"] = int(args.jobs)

    runner.run_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
