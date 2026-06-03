"""Ablation — per-conflict budget allocator (uniform vs risk-proportional).

Sweeps :func:`slackcertify.repair.algorithm.slack_certify`'s
``budget_alloc`` parameter across ``{uniform, risk_proportional}``
(see :mod:`slackcertify.repair.budget`). Probabilistic mode only:
the budget allocator does not apply in bounded mode.

``delta=1`` is fixed by the simulator-alignment convention from
Phase 7.2a/b (see :doc:`docs/algorithm.md` §"Aligning with the
simulator's collision model").

Expected finding: ``risk_proportional`` produces fewer total waits
than ``uniform`` when conflicts are imbalanced — it allocates budget
where it can actually be used. On balanced conflict sets the two
allocators degenerate.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from experiments.runners._base import CellResult, CellSpec, ExperimentRunner
from slackcertify.benchmarks.scenarios import ScenarioRegistry
from slackcertify.core.conflict import detect_conflicts
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.repair import (
    CertificationFailure,
    WaitInfeasibleError,
    slack_certify,
)
from slackcertify.simulate.rollout import monte_carlo_rollout
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import SolverError, SolverNotFoundError
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["AblationBudgetRunner"]


_VALID_ALLOCATORS = {"uniform", "risk_proportional"}


class AblationBudgetRunner(ExperimentRunner):
    """Ablation runner over the two probabilistic-mode budget allocators."""

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
                            delay_model_config={
                                "p_d": float(cfg.get("p_d", 0.03)),
                            },
                            rollouts=rollouts,
                        )

    def run_cell(self, cell_spec: CellSpec) -> CellResult:
        cfg = self.load_config()
        delta = int(cfg.get("delta", 1))
        result = CellResult(
            map_name=cell_spec.map_name,
            n_agents=cell_spec.n_agents,
            scen_seed=cell_spec.scen_seed,
            method=cell_spec.method,
        )

        if cell_spec.method not in _VALID_ALLOCATORS:
            result.status = "cert_failed"
            result.error_message = f"unknown allocator {cell_spec.method!r}"
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

        nominal_plan = self._solve_nominal(graph, agents, cfg, result)
        if nominal_plan is None:
            return result

        result.n_conflicts_initial = len(detect_conflicts(nominal_plan, delta=0))
        # Nominal-plan SoC — the denominator SoC(π) for the M2 SoC-overhead
        # metric SoC(π')/SoC(π) - 1. Captured before any wait insertion, using
        # the same sum-of-arrival-times definition as the executed SoC
        # (Plan.sum_of_costs == sum of per-agent arrival times).
        result.nominal_soc = float(nominal_plan.sum_of_costs)

        certified_plan = self._apply_allocator(nominal_plan, agents, cell_spec, cfg, delta, result)
        if certified_plan is None:
            return result

        p_d = float(cell_spec.delay_model_config["p_d"])
        self._run_rollouts(certified_plan, p_d, cell_spec, result)
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

    def _apply_allocator(
        self,
        nominal_plan: Plan,
        agents: list[Agent],
        cell_spec: CellSpec,
        cfg: dict[str, Any],
        delta: int,
        result: CellResult,
    ) -> Plan | None:
        allocator = cell_spec.method
        p_d = float(cfg.get("p_d", 0.03))
        epsilon = float(cfg.get("epsilon", 0.05))
        rounds_budget = max(50, len(agents) * 4)
        t0 = time.perf_counter()
        try:
            certified_plan, cert = slack_certify(
                nominal_plan,
                mode="probabilistic",
                p_d=p_d,
                epsilon=epsilon,
                delta=delta,
                budget_alloc=allocator,  # type: ignore[arg-type]
                max_outer_rounds=rounds_budget,
            )
        except WaitInfeasibleError as exc:
            result.status = "wait_infeasible"
            result.error_message = f"WaitInfeasibleError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        except CertificationFailure as exc:
            result.status = "cert_failed"
            result.error_message = f"CertificationFailure: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        except Exception as exc:  # noqa: BLE001
            result.status = "cert_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        result.certifier_runtime_s = time.perf_counter() - t0
        result.total_waits = cert.total_wait_inserted
        result.outer_rounds = cert.outer_rounds
        result.initial_unsafe_count = cert.initial_unsafe_count
        result.cumulative_unsafe_count = cert.cumulative_unsafe_count
        result.diagnostics["mode"] = "probabilistic"
        result.diagnostics["budget_alloc"] = allocator
        result.diagnostics["p_d"] = p_d
        result.diagnostics["epsilon"] = epsilon
        result.diagnostics["delta"] = delta
        return certified_plan

    def _run_rollouts(
        self,
        plan: Plan,
        p_d: float,
        cell_spec: CellSpec,
        result: CellResult,
    ) -> None:
        model = BernoulliDelayModel(p_d=p_d)
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
