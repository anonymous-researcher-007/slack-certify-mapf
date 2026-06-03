"""RQ1 — headline robustness comparison.

For each ``(map, n_agents, scen_seed)`` we solve the nominal MAPF
instance once (with LaCAM*, or :class:`FakeSolver` as a fallback when
the binary is missing) and then emit one cell per method:

* ``nominal_unhardened`` — the raw solver plan, rolled out under
  :class:`BoundedDelayModel`.
* ``slack_certify_bounded`` — Slack-Certify in bounded mode at the
  configured ``delta``, rolled out under the matching bounded model.
* ``slack_certify_probabilistic`` — Slack-Certify in probabilistic
  mode at the configured ``(p_d, epsilon)``, rolled out under the
  matching Bernoulli model.

Every certifier exception is caught and surfaced as a structured
status (``wait_infeasible`` / ``cert_failed`` / ``rollout_failed``)
in the resulting :class:`CellResult` row.
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
from slackcertify.delay.adversarial import HeuristicAdversaryDelay
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.simulate.rollout import (
    DelayModel,
    monte_carlo_rollout,
)
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import SolverError, SolverNotFoundError
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["RQ1HeadlineRunner"]


class RQ1HeadlineRunner(ExperimentRunner):
    """RQ1 headline runner — nominal vs. bounded vs. probabilistic SlackCertify."""

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
                            delay_model_config={},
                            rollouts=rollouts,
                        )

    def run_cell(self, cell_spec: CellSpec) -> CellResult:
        cfg = self.load_config()
        result = CellResult(
            map_name=cell_spec.map_name,
            n_agents=cell_spec.n_agents,
            scen_seed=cell_spec.scen_seed,
            method=cell_spec.method,
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

        plan_to_test = self._apply_method(nominal_plan, agents, cell_spec, cfg, result)
        if plan_to_test is None:
            return result  # certifier raised; status already set

        delay_model = self._build_delay_model(cell_spec.method, cfg)
        self._run_rollouts(plan_to_test, delay_model, cell_spec, result)
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
        agents: list[Agent],
        cell_spec: CellSpec,
        cfg: dict[str, Any],
        result: CellResult,
    ) -> Plan | None:
        method = cell_spec.method
        if method == "nominal_unhardened":
            return nominal_plan

        t0 = time.perf_counter()
        diag: dict[str, Any]
        try:
            if method == "slack_certify_bounded":
                bounded_cfg = cfg["slack_certify_bounded"]
                delta = int(bounded_cfg["delta"])
                certified_plan, diag = stn_certify(
                    nominal_plan, delta=delta, mode="bounded", return_diagnostics=True
                )
                result.diagnostics["mode"] = "bounded"
                result.diagnostics["delta"] = delta
            elif method == "slack_certify_probabilistic":
                prob_cfg = cfg["slack_certify_probabilistic"]
                p_d = float(prob_cfg["p_d"])
                epsilon = float(prob_cfg["epsilon"])
                certified_plan, diag = stn_certify(
                    nominal_plan,
                    mode="probabilistic",
                    p_d=p_d,
                    epsilon=epsilon,
                    return_diagnostics=True,
                )
                result.diagnostics["mode"] = "probabilistic"
                result.diagnostics["p_d"] = p_d
                result.diagnostics["epsilon"] = epsilon
            else:
                result.status = "cert_failed"
                result.error_message = f"unknown method {method!r}"
                return None
        except STNInfeasible as exc:
            # No fixed-order wait-only repair exists (e.g. rotational deadlock /
            # probabilistic s-search exhausted). A legitimate, recorded outcome
            # — not a crash.
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
        return certified_plan

    def _build_delay_model(self, method: str, cfg: dict[str, Any]) -> DelayModel:
        # delay_protocol is the §V P1 knob: "bernoulli" (default) keeps the
        # existing i.i.d. rollout; "adversarial" engages the heuristic
        # per-tick gap-collapsing adversary under the bounded-Δ budget.
        delay_protocol = str(cfg.get("delay_protocol", "bernoulli")).lower()
        if delay_protocol == "adversarial":
            delta = int(cfg["slack_certify_bounded"]["delta"])
            return HeuristicAdversaryDelay(delta=delta)
        if method == "slack_certify_probabilistic":
            p_d = float(cfg["slack_certify_probabilistic"]["p_d"])
            return BernoulliDelayModel(p_d=p_d)
        delta = int(cfg["slack_certify_bounded"]["delta"])
        return BoundedDelayModel(delta=delta)

    def _run_rollouts(
        self,
        plan: Plan,
        delay_model: DelayModel,
        cell_spec: CellSpec,
        result: CellResult,
    ) -> None:
        seed = derive_seed(cell_spec.scen_seed, cell_spec.method)
        rng = np.random.default_rng(seed)
        try:
            rollout = monte_carlo_rollout(plan, delay_model, K=cell_spec.rollouts, rng=rng)
        except Exception as exc:  # noqa: BLE001
            result.status = "rollout_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            return

        result.success_rate = rollout.success_rate
        result.success_rate_ci_lo, result.success_rate_ci_hi = rollout.wilson_ci_95
        result.executed_soc_mean = rollout.mean_executed_soc
        result.executed_makespan_mean = rollout.mean_executed_makespan
        result.status = "ok"
