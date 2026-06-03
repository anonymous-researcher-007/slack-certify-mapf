"""Ablation — nominal-solver front-end (EECBS / LaCAM* / PIBT).

Sweeps the nominal MAPF solver to demonstrate SlackCertify is
solver-agnostic: certifier metrics (total_waits, outer_rounds,
runtime) should be roughly invariant across front-ends modulo the
nominal plan's conflict density.

When a solver binary is missing the cell is recorded with
``status="binary_missing"`` (no fallback to FakeSolver — the whole
point of this ablation is to compare the three real solvers). The
Phase 8 analysis pipeline drops binary_missing rows from
comparisons.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from experiments.runners._base import (
    CellResult,
    CellSpec,
    ExperimentRunner,
    record_stn_diagnostics,
)
from slackcertify.benchmarks.scenarios import ScenarioRegistry
from slackcertify.core.conflict import detect_conflicts
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.simulate.rollout import monte_carlo_rollout
from slackcertify.solvers.base import (
    BinarySolverBase,
    SolverError,
    SolverNotFoundError,
    SolverTimeoutError,
)
from slackcertify.solvers.eecbs import EECBSSolver
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.solvers.pibt import PIBTSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["AblationSolverRunner"]


_SOLVER_CLASSES: dict[str, type[BinarySolverBase]] = {
    "eecbs": EECBSSolver,
    "lacam_star": LaCAMStarSolver,
    "pibt": PIBTSolver,
}


class AblationSolverRunner(ExperimentRunner):
    """Ablation runner over the three external MAPF solver front-ends."""

    def enumerate_cells(self) -> Iterator[CellSpec]:
        cfg = self.load_config()
        rollouts = int(cfg["rollouts"])
        for map_name in cfg["maps"]:
            for n_agents in cfg["agent_counts"]:
                for scen_seed in cfg["scen_seeds"]:
                    for method in cfg["methods"]:
                        yield CellSpec(
                            map_name=str(map_name),
                            n_agents=int(n_agents),
                            scen_seed=int(scen_seed),
                            method=str(method),
                            delay_model_config={"delta": int(cfg.get("delta", 2))},
                            rollouts=rollouts,
                        )

    def run_cell(self, cell_spec: CellSpec) -> CellResult:
        cfg = self.load_config()
        delta = int(cfg.get("delta", 2))
        result = CellResult(
            map_name=cell_spec.map_name,
            n_agents=cell_spec.n_agents,
            scen_seed=cell_spec.scen_seed,
            method=cell_spec.method,
        )

        if cell_spec.method not in _SOLVER_CLASSES:
            result.status = "cert_failed"
            result.error_message = f"unknown solver {cell_spec.method!r}"
            return result

        registry = (
            ScenarioRegistry(root=self.benchmarks_root)
            if self.benchmarks_root is not None
            else ScenarioRegistry()
        )
        try:
            graph, agents = registry.load_instance(
                cell_spec.map_name, cell_spec.n_agents, cell_spec.scen_seed
            )
        except FileNotFoundError as exc:
            result.status = "nominal_failed"
            result.error_message = f"FileNotFoundError: {exc}"
            return result

        nominal_plan = self._solve_nominal(graph, agents, cfg, cell_spec, result)
        if nominal_plan is None:
            return result

        result.n_conflicts_initial = len(detect_conflicts(nominal_plan, delta=0))
        # Nominal-plan SoC — the denominator SoC(π) for the M2 SoC-overhead
        # metric SoC(π')/SoC(π) - 1. Captured before any wait insertion, using
        # the same sum-of-arrival-times definition as the executed SoC
        # (Plan.sum_of_costs == sum of per-agent arrival times).
        result.nominal_soc = float(nominal_plan.sum_of_costs)
        result.diagnostics["solver"] = cell_spec.method

        certified_plan = self._apply_certify(nominal_plan, agents, delta, result)
        if certified_plan is None:
            return result

        self._run_rollouts(certified_plan, delta, cell_spec, result)
        return result

    # --------------------------------------------------------------- helpers

    def _solve_nominal(
        self,
        graph: GridGraph,
        agents: list[Agent],
        cfg: dict[str, Any],
        cell_spec: CellSpec,
        result: CellResult,
    ) -> Plan | None:
        time_limit = float(cfg.get("nominal_time_limit_s", 10.0))
        solver_cls = _SOLVER_CLASSES[cell_spec.method]
        t0 = time.perf_counter()
        try:
            solver = solver_cls()
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: {cell_spec.method} binary "
                f"missing ({exc}); recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.nominal_solver_runtime_s = time.perf_counter() - t0
            return None
        try:
            nominal_plan = solver.solve(graph, agents, time_limit_s=time_limit)
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: {cell_spec.method} binary "
                f"missing ({exc}); recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.nominal_solver_runtime_s = time.perf_counter() - t0
            return None
        except SolverTimeoutError as exc:
            result.status = "timeout"
            result.error_message = f"SolverTimeoutError: {exc}"
            result.nominal_solver_runtime_s = time.perf_counter() - t0
            return None
        except SolverError as exc:
            result.status = "nominal_failed"
            result.error_message = f"SolverError: {exc}"
            result.nominal_solver_runtime_s = time.perf_counter() - t0
            return None
        except Exception as exc:  # noqa: BLE001
            result.status = "nominal_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.nominal_solver_runtime_s = time.perf_counter() - t0
            return None
        result.nominal_solver_runtime_s = time.perf_counter() - t0
        return nominal_plan

    def _apply_certify(
        self,
        nominal_plan: Plan,
        agents: list[Agent],
        delta: int,
        result: CellResult,
    ) -> Plan | None:
        t0 = time.perf_counter()
        try:
            certified_plan, diag = stn_certify(
                nominal_plan, delta=delta, mode="bounded", return_diagnostics=True
            )
        except STNInfeasible as exc:
            result.status = "wait_infeasible"
            result.error_message = f"STNInfeasible[{exc.reason}]: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            result.diagnostics["stn_infeasible"] = True
            result.diagnostics["stn_infeasible_reason"] = exc.reason
            return None
        except Exception as exc:  # noqa: BLE001
            result.status = "cert_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        result.certifier_runtime_s = time.perf_counter() - t0
        record_stn_diagnostics(result, diag)
        result.diagnostics["mode"] = "bounded"
        result.diagnostics["delta"] = delta
        return certified_plan

    def _run_rollouts(
        self,
        plan: Plan,
        delta: int,
        cell_spec: CellSpec,
        result: CellResult,
    ) -> None:
        model = BoundedDelayModel(delta=delta)
        seed = derive_seed(cell_spec.scen_seed, f"{cell_spec.method}/rollout")
        rng = np.random.default_rng(seed)
        try:
            rollout = monte_carlo_rollout(plan, model, K=cell_spec.rollouts, rng=rng)
        except Exception as exc:  # noqa: BLE001
            result.status = "rollout_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            return
        result.success_rate = rollout.success_rate
        result.success_rate_ci_lo, result.success_rate_ci_hi = rollout.wilson_ci_95
        result.executed_soc_mean = rollout.mean_executed_soc
        result.executed_makespan_mean = rollout.mean_executed_makespan
        result.status = "ok"
