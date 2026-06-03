"""RQ2 — k-Robust Pareto sweep.

For each ``(map, n_agents, scen_seed, delta)`` we solve the nominal MAPF
instance once (with LaCAM*, or :class:`FakeSolver` as a fallback when
the binary is missing) and then emit one cell per method:

* ``slack_certify_bounded`` — Slack-Certify in bounded mode at the
  cell's ``delta``.
* ``kr_eecbs`` — k-Robust EECBS at ``k = delta``.
* ``kr_pbs`` — k-Robust PBS at ``k = delta``.

The k-Robust baselines require external binaries. When the binary is
missing we record ``status = "binary_missing"`` (with a log warning)
and emit no further work for that cell; the Phase 8 analysis pipeline
drops such rows from comparisons.

Rollouts always execute under :class:`BoundedDelayModel(delta=delta)`.
We additionally record the worst-case adversarial schedule's collision
outcome (a single deterministic execution) in
``diagnostics["worst_case_collision"]`` so the analysis layer can read
both the average and the worst-case behaviour off one row.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from baselines.kr_cbs.wrapper import KRobustCBSBaseline, KRobustPBSBaseline

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
from slackcertify.delay.adversarial import HeuristicAdversaryDelay
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.simulate.executor import Executor
from slackcertify.simulate.rollout import DelayModel, monte_carlo_rollout
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import (
    SolverError,
    SolverNotFoundError,
    SolverTimeoutError,
)
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["RQ2KRParetoRunner"]


class RQ2KRParetoRunner(ExperimentRunner):
    """RQ2 runner — Slack-Certify bounded vs. k-Robust EECBS/PBS at swept ``delta``."""

    def enumerate_cells(self) -> Iterator[CellSpec]:
        cfg = self.load_config()
        rollouts = int(cfg["rollouts"])
        for map_name in cfg["maps"]:
            for n_agents in cfg["agent_counts"]:
                for scen_seed in cfg["scen_seeds"]:
                    for delta in cfg["delta_sweep"]:
                        for method in cfg["methods"]:
                            yield CellSpec(
                                map_name=str(map_name),
                                n_agents=int(n_agents),
                                scen_seed=int(scen_seed),
                                method=str(method),
                                delay_model_config={"delta": int(delta)},
                                rollouts=rollouts,
                                delta=int(delta),
                            )

    def run_cell(self, cell_spec: CellSpec) -> CellResult:
        cfg = self.load_config()
        delta = int(cell_spec.delta) if cell_spec.delta is not None else 0
        result = CellResult(
            map_name=cell_spec.map_name,
            n_agents=cell_spec.n_agents,
            scen_seed=cell_spec.scen_seed,
            method=cell_spec.method,
            delta=delta,
        )

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

        nominal_plan = self._solve_nominal(graph, agents, cfg, result)
        if nominal_plan is None:
            return result

        result.n_conflicts_initial = len(detect_conflicts(nominal_plan, delta=0))
        # Nominal-plan SoC — the denominator SoC(π) for the M2 SoC-overhead
        # metric SoC(π')/SoC(π) - 1. Captured before any wait insertion, using
        # the same sum-of-arrival-times definition as the executed SoC
        # (Plan.sum_of_costs == sum of per-agent arrival times).
        result.nominal_soc = float(nominal_plan.sum_of_costs)

        plan_to_test = self._apply_method(nominal_plan, graph, agents, cell_spec, delta, result)
        if plan_to_test is None:
            return result  # status already set

        self._run_rollouts(plan_to_test, nominal_plan, cell_spec, delta, cfg, result)
        return result

    # --------------------------------------------------------------- helpers

    def _solve_nominal(
        self,
        graph: GridGraph,
        agents: list[Agent],
        cfg: dict[str, Any],
        result: CellResult,
    ) -> Plan | None:
        time_limit = float(cfg.get("nominal_time_limit_s", 10.0))
        t0 = time.perf_counter()
        try:
            nominal_plan = LaCAMStarSolver().solve(graph, agents, time_limit_s=time_limit)
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: nominal LaCAM* binary missing "
                f"({exc}); falling back to FakeSolver"
            )
            result.diagnostics["solver_fallback"] = True
            try:
                nominal_plan = FakeSolver().solve(graph, agents, time_limit_s=time_limit)
            except (SolverError, Exception) as exc2:  # noqa: BLE001
                result.status = "nominal_failed"
                result.error_message = f"{type(exc2).__name__}: {exc2}"
                result.nominal_solver_runtime_s = time.perf_counter() - t0
                return None
        except Exception as exc:  # noqa: BLE001
            result.status = "nominal_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.nominal_solver_runtime_s = time.perf_counter() - t0
            return None
        result.nominal_solver_runtime_s = time.perf_counter() - t0
        return nominal_plan

    def _apply_method(
        self,
        nominal_plan: Plan,
        graph: GridGraph,
        agents: list[Agent],
        cell_spec: CellSpec,
        delta: int,
        result: CellResult,
    ) -> Plan | None:
        method = cell_spec.method
        t0 = time.perf_counter()
        try:
            if method == "slack_certify_bounded":
                certified_plan, diag = stn_certify(
                    nominal_plan, delta=delta, mode="bounded", return_diagnostics=True
                )
                result.diagnostics["mode"] = "bounded"
                result.diagnostics["delta"] = delta
                result.certifier_runtime_s = time.perf_counter() - t0
                record_stn_diagnostics(result, diag)
                return certified_plan
            if method in ("kr_eecbs", "kr_pbs"):
                return self._solve_k_robust(method, graph, agents, delta, result, t0)
            result.status = "cert_failed"
            result.error_message = f"unknown method {method!r}"
            return None
        except STNInfeasible as exc:
            result.status = "wait_infeasible"
            result.error_message = f"STNInfeasible[{exc.reason}]: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            result.diagnostics["stn_infeasible"] = True
            result.diagnostics["stn_infeasible_reason"] = exc.reason
            return None
        except SolverTimeoutError as exc:
            result.status = "timeout"
            result.error_message = f"SolverTimeoutError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        except Exception as exc:  # noqa: BLE001
            result.status = "cert_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None

    def _solve_k_robust(
        self,
        method: str,
        graph: GridGraph,
        agents: list[Agent],
        delta: int,
        result: CellResult,
        t0: float,
    ) -> Plan | None:
        baseline_cls = KRobustCBSBaseline if method == "kr_eecbs" else KRobustPBSBaseline
        try:
            baseline = baseline_cls()
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: {method} binary missing ({exc}); "
                f"recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        try:
            baseline_plan = baseline.solve(graph, agents, k=delta)
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: {method} binary missing ({exc}); "
                f"recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        result.diagnostics["mode"] = method
        result.diagnostics["k"] = delta
        result.diagnostics["solver_used"] = baseline_plan.solver_used
        result.certifier_runtime_s = time.perf_counter() - t0
        return baseline_plan.plan

    def _run_rollouts(
        self,
        plan: Plan,
        nominal_plan: Plan,
        cell_spec: CellSpec,
        delta: int,
        cfg: dict[str, Any],
        result: CellResult,
    ) -> None:
        # delay_protocol is the §V P1 knob: "bounded" (default) keeps the
        # existing random feasible-Δ rollout; "adversarial" engages the
        # heuristic per-tick gap-collapsing adversary under the same Δ.
        delay_protocol = str(cfg.get("delay_protocol", "bounded")).lower()
        model: DelayModel = (
            HeuristicAdversaryDelay(delta=delta)
            if delay_protocol == "adversarial"
            else BoundedDelayModel(delta=delta)
        )
        seed = derive_seed(cell_spec.scen_seed, f"{cell_spec.method}/delta={delta}")
        rng = np.random.default_rng(seed)
        try:
            rollout = monte_carlo_rollout(plan, model, K=cell_spec.rollouts, rng=rng)
        except Exception as exc:  # noqa: BLE001
            result.status = "rollout_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            return

        # Worst-case adversarial schedule: deterministic single execution
        # against the certifier's view of the nominal conflicts. Recorded
        # so the analysis layer can compare against the MC average.
        try:
            adv_conflicts = detect_conflicts(nominal_plan, delta=delta)
            adv_schedule = model.worst_case_against(plan, adv_conflicts)
            adv_trace = Executor(plan, adv_schedule).run()
            result.diagnostics["worst_case_collision"] = bool(adv_trace.collisions)
        except Exception as exc:  # noqa: BLE001 - diagnostic-only path
            result.diagnostics["worst_case_collision"] = None
            result.diagnostics["worst_case_error"] = f"{type(exc).__name__}: {exc}"

        result.success_rate = rollout.success_rate
        result.success_rate_ci_lo, result.success_rate_ci_hi = rollout.wilson_ci_95
        result.executed_soc_mean = rollout.mean_executed_soc
        result.executed_makespan_mean = rollout.mean_executed_makespan
        result.status = "ok"
