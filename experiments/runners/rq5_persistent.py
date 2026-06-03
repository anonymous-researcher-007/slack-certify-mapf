"""RQ5 — persistent (MAPF-DP-style) delay stress test.

Same scaffolding as :mod:`rq1_headline`, but the delay model is
:class:`PersistentDelayModel` and the probabilistic SlackCertify is
parameterised by the *calibrated* ``p_d`` (the empirical per-step
blocking rate of the persistent model), recorded under
``diagnostics["calibrated_p_d"]`` for traceability.
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
from slackcertify.delay.calibration import calibrate_bernoulli_to_persistent
from slackcertify.delay.persistent import PersistentDelayModel
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.simulate.rollout import monte_carlo_rollout
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import SolverError, SolverNotFoundError
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["RQ5PersistentRunner"]


class RQ5PersistentRunner(ExperimentRunner):
    """RQ5 persistent runner — nominal vs. probabilistic SlackCertify under MAPF-DP delays."""

    def enumerate_cells(self) -> Iterator[CellSpec]:
        cfg = self.load_config()
        rollouts = int(cfg["rollouts"])
        # Optional certificate-tightness sweep (RQ5): one cell per target
        # epsilon. Absent ``epsilon_sweep`` ⇒ a single [None] pass, leaving
        # the runner's pre-sweep behaviour (epsilon read from the config
        # block) unchanged. The per-cell epsilon enters CellSpec.key() so
        # cells differing only in epsilon resume independently.
        eps_sweep = cfg.get("epsilon_sweep")
        epsilons: list[float | None] = (
            [float(e) for e in eps_sweep] if eps_sweep else [None]
        )
        for map_name in cfg["maps"]:
            for n_agents in cfg["agent_counts"]:
                for scen_seed in cfg["scen_seeds"]:
                    for method in cfg["methods"]:
                        for epsilon in epsilons:
                            yield CellSpec(
                                map_name=str(map_name),
                                n_agents=int(n_agents),
                                scen_seed=int(scen_seed),
                                method=str(method),
                                delay_model_config=dict(cfg["persistent"]),
                                rollouts=rollouts,
                                epsilon=epsilon,
                            )

    def run_cell(self, cell_spec: CellSpec) -> CellResult:
        cfg = self.load_config()
        result = CellResult(
            map_name=cell_spec.map_name,
            n_agents=cell_spec.n_agents,
            scen_seed=cell_spec.scen_seed,
            method=cell_spec.method,
        )
        # Record the cell's swept epsilon coordinate (None for non-sweep
        # runs). This MUST mirror CellSpec.epsilon — it is the value the
        # manifest stores and resume matches on, so it has to equal the key
        # coordinate, not the effective certifier epsilon. The effective
        # epsilon used by the certifier (with the config-block fallback) is
        # recorded separately in diagnostics["epsilon"]. On every row
        # (including nominal / failures) this lets the tightness analysis
        # plot empirical_collision_rate vs. epsilon.
        result.epsilon = cell_spec.epsilon

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

        # Build the persistent delay model once per cell (deterministic).
        persistent = PersistentDelayModel(
            p_trigger=float(cell_spec.delay_model_config["p_trigger"]),
            min_len=int(cell_spec.delay_model_config["min_len"]),
            max_len=int(cell_spec.delay_model_config["max_len"]),
        )

        # Apply method.
        plan_to_test = self._apply_method(nominal_plan, agents, cell_spec, cfg, persistent, result)
        if plan_to_test is None:
            return result

        # Always roll out under the persistent model.
        seed = derive_seed(cell_spec.scen_seed, f"{cell_spec.method}/rollout")
        rng = np.random.default_rng(seed)
        try:
            rollout = monte_carlo_rollout(plan_to_test, persistent, K=cell_spec.rollouts, rng=rng)
        except Exception as exc:  # noqa: BLE001
            result.status = "rollout_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            return result

        result.success_rate = rollout.success_rate
        result.success_rate_ci_lo, result.success_rate_ci_hi = rollout.wilson_ci_95
        result.executed_soc_mean = rollout.mean_executed_soc
        result.executed_makespan_mean = rollout.mean_executed_makespan
        # Empirical collision rate (metric M9): the fraction of rollouts in
        # which ANY collision occurred. Computed directly from the per-rollout
        # collision counts rather than assuming the 1 - success_rate identity
        # (a rollout's success is defined as "zero collisions", so the two
        # agree, but we derive it from the raw counts for robustness).
        per_rollout = rollout.n_collisions_per_rollout
        n_rollouts = len(per_rollout)
        n_collided = sum(1 for c in per_rollout if c > 0)
        result.empirical_collision_rate = (
            n_collided / n_rollouts if n_rollouts else float("nan")
        )
        result.status = "ok"
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
        persistent: PersistentDelayModel,
        result: CellResult,
    ) -> Plan | None:
        method = cell_spec.method
        if method == "nominal_unhardened":
            return nominal_plan

        if method != "slack_certify_probabilistic":
            result.status = "cert_failed"
            result.error_message = f"unknown method {method!r}"
            return None

        # Calibrate Bernoulli p_d to the persistent model.
        plan_length = max((p.makespan for p in nominal_plan.paths), default=1)
        calibration_rollouts = int(cfg.get("calibration_rollouts", 200))
        cal_rng = np.random.default_rng(derive_seed(cell_spec.scen_seed, "rq5_calibration"))
        # `calibrate_bernoulli_to_persistent` always uses 200 rollouts; we
        # honour the config value here for traceability even though the
        # function ignores it.
        _ = calibration_rollouts
        calibrated_p_d = float(
            calibrate_bernoulli_to_persistent(persistent, plan_length=plan_length, rng=cal_rng)
        )
        # Clamp into Bernoulli model's domain [0, 1).
        if calibrated_p_d >= 1.0:
            calibrated_p_d = 0.999999

        result.diagnostics["mode"] = "probabilistic"
        result.diagnostics["calibrated_p_d"] = calibrated_p_d
        result.diagnostics["calibration_plan_length"] = plan_length

        # Prefer the cell's swept epsilon (RQ5 tightness); fall back to the
        # config block's epsilon for non-sweep runs.
        epsilon = (
            float(cell_spec.epsilon)
            if cell_spec.epsilon is not None
            else float(cfg["slack_certify_probabilistic"]["epsilon"])
        )
        t0 = time.perf_counter()
        try:
            certified_plan, diag = stn_certify(
                nominal_plan,
                mode="probabilistic",
                p_d=calibrated_p_d,
                epsilon=epsilon,
                return_diagnostics=True,
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
        result.diagnostics["epsilon"] = epsilon
        return certified_plan
