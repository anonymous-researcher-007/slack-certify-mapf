"""Experiment-runner base class.

A ``Runner`` owns a YAML config, a CSV result file, and a JSONL manifest
file. It enumerates *cells* (one row of the experiment grid), runs each
one, and **appends** the result to both files in a crash-safe way:

* CSV rows are appended via the OS's ``O_APPEND`` semantics and an
  explicit ``fsync`` after every row, so a crash mid-run drops at most
  the in-progress row.
* The JSONL manifest is updated the same way and is consulted on the
  next run to skip cells that already finished. This makes long runs
  (24–72 h for the paper-quality grids) restartable.

The :func:`run_all` driver catches every per-cell exception and records
it with a non-``"ok"`` status; it never crashes the run as a whole.
"""

from __future__ import annotations

import csv
import json
import os
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from rich.console import Console

__all__ = [
    "CSV_FIELDS",
    "CellResult",
    "CellSpec",
    "ExperimentRunner",
    "Status",
    "record_stn_diagnostics",
]


def record_stn_diagnostics(result: CellResult, diag: dict[str, Any]) -> None:
    """Record STN certifier diagnostics on a successful certification.

    Populates ``total_waits`` (an existing column) and the free-form
    ``diagnostics`` blob with the separation parameter ``s``, the solver
    path, and — in probabilistic mode — the binary-search iteration count.
    These replace the old greedy round-count / slack-ratio columns
    (``outer_rounds`` / ``initial_unsafe_count`` / ``cumulative_unsafe_count``
    stay at their defaults for STN-certified cells).
    """
    result.total_waits = int(diag["total_waits"])
    result.diagnostics["stn_infeasible"] = False
    result.diagnostics["stn_separation_s"] = diag.get("separation_s")
    result.diagnostics["stn_solver"] = diag.get("solver")
    if "search_iters" in diag:
        result.diagnostics["stn_search_iters"] = diag["search_iters"]


Status = Literal[
    "ok",
    "wait_infeasible",
    "cert_failed",
    "rollout_failed",
    "nominal_failed",
    "timeout",
    "binary_missing",
]


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One row of the experiment grid.

    ``delta`` is optional and only set by runners (e.g. RQ2) that sweep
    a bounded-delay margin per cell; when present it participates in the
    manifest key so cells differing only in ``delta`` are tracked
    independently. ``epsilon`` plays the same role for runners (e.g. the
    RQ5 certificate-tightness sweep) that sweep the probabilistic target
    risk per cell.
    """

    map_name: str
    n_agents: int
    scen_seed: int
    method: str
    delay_model_config: dict[str, Any]
    rollouts: int
    delta: int | None = None
    epsilon: float | None = None

    def key(self) -> tuple[str, int, int, str, int | None, float | None]:
        """Manifest key — uniquely identifies the cell."""
        return (
            self.map_name,
            self.n_agents,
            self.scen_seed,
            self.method,
            self.delta,
            self.epsilon,
        )


@dataclass
class CellResult:
    """Single-row output of one cell."""

    map_name: str
    n_agents: int
    scen_seed: int
    method: str
    success_rate: float = float("nan")
    success_rate_ci_lo: float = float("nan")
    success_rate_ci_hi: float = float("nan")
    empirical_collision_rate: float = float("nan")
    nominal_soc: float = float("nan")
    executed_soc_mean: float = float("nan")
    executed_makespan_mean: float = float("nan")
    certifier_runtime_s: float = 0.0
    nominal_solver_runtime_s: float = 0.0
    total_waits: int = 0
    outer_rounds: int = 0
    initial_unsafe_count: int = 0
    cumulative_unsafe_count: int = 0
    n_conflicts_initial: int = 0
    delta: int | None = None
    epsilon: float | None = None
    status: Status = "ok"
    error_message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


CSV_FIELDS: list[str] = [
    "map_name",
    "n_agents",
    "scen_seed",
    "method",
    "success_rate",
    "success_rate_ci_lo",
    "success_rate_ci_hi",
    "empirical_collision_rate",
    "nominal_soc",
    "executed_soc_mean",
    "executed_makespan_mean",
    "certifier_runtime_s",
    "nominal_solver_runtime_s",
    "total_waits",
    "outer_rounds",
    "initial_unsafe_count",
    "cumulative_unsafe_count",
    "n_conflicts_initial",
    "delta",
    "epsilon",
    "status",
    "error_message",
    "diagnostics",
]


class ExperimentRunner(ABC):
    """Base class for the §V experiment runners.

    Parameters
    ----------
    config_path
        YAML config (sweep dimensions, method list, hyperparameters).
    output_csv
        CSV where each completed cell appends one row.
    manifest_path
        JSONL log used to skip already-completed cells on restart.
    benchmarks_root
        Optional override for the MovingAI benchmark root; defaults
        to ``<repo>/benchmarks/``. Tests pass a fixture path here.
    """

    def __init__(
        self,
        config_path: str | Path,
        output_csv: str | Path,
        manifest_path: str | Path,
        benchmarks_root: str | Path | None = None,
    ) -> None:
        self.config_path: Path = Path(config_path)
        self.output_csv: Path = Path(output_csv)
        self.manifest_path: Path = Path(manifest_path)
        self.benchmarks_root: Path | None = (
            Path(benchmarks_root) if benchmarks_root is not None else None
        )
        self._config: dict[str, Any] | None = None
        self.console: Console = Console()

    # --------------------------------------------------------------- config

    def load_config(self) -> dict[str, Any]:
        """Load (and cache) the YAML config."""
        if self._config is None:
            with self.config_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if not isinstance(loaded, dict):
                raise ValueError(f"{self.config_path}: expected a YAML mapping at the top level")
            self._config = loaded
        return self._config

    # ------------------------------------------------------------- manifest

    def resume_from_manifest(
        self,
    ) -> set[tuple[str, int, int, str, int | None, float | None]]:
        """Return the set of cell keys already recorded in the manifest."""
        completed: set[tuple[str, int, int, str, int | None, float | None]] = set()
        if not self.manifest_path.exists():
            return completed
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Half-written line from a previous crash; ignore.
                    continue
                try:
                    delta_raw = obj.get("delta")
                    delta_val: int | None = None if delta_raw is None else int(delta_raw)
                    # epsilon is absent in pre-sweep manifests; default None so
                    # the key matches a CellSpec that also leaves epsilon unset.
                    eps_raw = obj.get("epsilon")
                    eps_val: float | None = None if eps_raw is None else float(eps_raw)
                    key = (
                        str(obj["map_name"]),
                        int(obj["n_agents"]),
                        int(obj["scen_seed"]),
                        str(obj["method"]),
                        delta_val,
                        eps_val,
                    )
                except (KeyError, ValueError, TypeError):
                    continue
                completed.add(key)
        return completed

    # ------------------------------------------------------------- abstract

    @abstractmethod
    def enumerate_cells(self) -> Iterator[CellSpec]:
        """Yield every cell in this experiment's sweep."""

    @abstractmethod
    def run_cell(self, cell_spec: CellSpec) -> CellResult:
        """Execute one cell and return its (possibly error) result."""

    # ---------------------------------------------------------- append + run

    def append_result(self, result: CellResult) -> None:
        """Append one CSV row and one manifest line, fsyncing each."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            field: getattr(result, field) for field in CSV_FIELDS if field != "diagnostics"
        }
        row["diagnostics"] = json.dumps(result.diagnostics, default=str)

        is_new = not self.output_csv.exists() or self.output_csv.stat().st_size == 0
        with self.output_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())

        manifest_line = json.dumps(
            {
                "map_name": result.map_name,
                "n_agents": result.n_agents,
                "scen_seed": result.scen_seed,
                "method": result.method,
                "delta": result.delta,
                "epsilon": result.epsilon,
                "status": result.status,
            }
        )
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(manifest_line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def run_all(self) -> None:
        """Iterate every cell, skipping ones already in the manifest."""
        completed = self.resume_from_manifest()
        for cell in self.enumerate_cells():
            if cell.key() in completed:
                self.console.log(f"skip {cell.key()} (already in manifest)")
                continue
            try:
                result = self.run_cell(cell)
            except Exception as exc:  # noqa: BLE001 - per-cell isolation
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                result = CellResult(
                    map_name=cell.map_name,
                    n_agents=cell.n_agents,
                    scen_seed=cell.scen_seed,
                    method=cell.method,
                    delta=cell.delta,
                    status="rollout_failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                    diagnostics={"traceback": tb},
                )
            self.append_result(result)
            self.console.log(
                f"cell {cell.key()}: status={result.status} "
                f"success_rate={result.success_rate:.4f}"
            )
