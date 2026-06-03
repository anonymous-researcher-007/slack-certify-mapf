"""RQ3 — close-analogue baselines under Bernoulli per-step delays.

Compares Slack-Certify (probabilistic) against two close-analogue
post-hoc / proactive baselines:

* ``slack_certify_probabilistic`` — Slack-Certify in probabilistic
  mode at ``delta=1, p_d=0.03, epsilon=0.05``. The ``delta=1`` matches
  the simulator's any-integer-tick occupation-window overlap (see
  ``docs/algorithm.md`` §"Aligning with the simulator's collision
  model").
* ``btpg_max`` — BTPG-max post-processing relaxation of Type-2 TPG
  edges (Su et al., AAAI 2024). External binary; ``binary_missing``
  if not built.
* ``kottinger_offline`` — the upstream Kottinger ``lns`` binary
  (aria-systems-group/Delay-Robust-MAPF). It is instance-matched: given
  the SAME map+scen as the nominal solve it computes its OWN nominal LNS
  plan, injects ``--numDelays`` delay events (default 1, Kottinger's
  headline single-delay regime), repairs on the constrained graph, and
  reports aggregate SoC/runtime/delay stats via CSV. RQ3 records those
  self-reported metrics directly rather than rolling out a reconstructed
  plan (the LNS CSV has no per-agent paths, and re-simulating would mix
  its discrete-delay semantics with the Bernoulli model). See
  ``baselines/kottinger/wrapper.py``.

Rollouts (for the slack_certify and btpg_max methods) execute under
:class:`BernoulliDelayModel(p_d=0.03)`. The kottinger_offline method does
not roll out; it reports Kottinger's own constrained-graph repair SoC.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from baselines.btpg.wrapper import BTPGMaxBaseline
from baselines.kottinger.wrapper import KottingerDelayBaseline

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
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.simulate.rollout import monte_carlo_rollout
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import (
    SolverError,
    SolverNotFoundError,
    SolverTimeoutError,
)
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["RQ3CloseAnaloguesRunner"]


class RQ3CloseAnaloguesRunner(ExperimentRunner):
    """RQ3 runner — Slack-Certify probabilistic vs. BTPG-max and Kottinger-offline."""

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
                            delay_model_config=dict(cfg["bernoulli"]),
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

        # Kottinger is an instance-matched baseline that solves the instance
        # itself and reports its OWN metrics (its nominal SoC, its repaired
        # constrained-graph SoC, its delay counts) via CSV — it does not
        # return a rollout-able Plan. So it bypasses the apply-method →
        # Monte-Carlo-rollout path and records its self-reported numbers
        # directly. See baselines/kottinger/wrapper.py and the RQ3 brief
        # (STEP 2/STEP 4).
        if cell_spec.method == "kottinger_offline":
            self._run_kottinger_offline(graph, agents, cell_spec, result)
            return result

        plan_to_test = self._apply_method(nominal_plan, agents, cell_spec, cfg, result)
        if plan_to_test is None:
            return result  # status already set

        p_d = float(cell_spec.delay_model_config["p_d"])
        self._run_rollouts(plan_to_test, p_d, cell_spec, result)
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
        t0 = time.perf_counter()
        try:
            if method == "slack_certify_probabilistic":
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
                result.certifier_runtime_s = time.perf_counter() - t0
                record_stn_diagnostics(result, diag)
                return certified_plan
            if method == "btpg_max":
                return self._apply_btpg_max(nominal_plan, result, t0)
            # kottinger_offline is handled in run_cell (it reports its own
            # metrics and is not rolled out), so it never reaches here.
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

    def _apply_btpg_max(self, nominal_plan: Plan, result: CellResult, t0: float) -> Plan | None:
        try:
            baseline = BTPGMaxBaseline()
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: btpg_max binary missing ({exc}); "
                f"recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        try:
            baseline_plan = baseline.post_process(nominal_plan)
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: btpg_max binary missing ({exc}); "
                f"recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None
        result.diagnostics["mode"] = "btpg_max"
        result.diagnostics["solver_used"] = baseline_plan.solver_used
        result.certifier_runtime_s = time.perf_counter() - t0
        return baseline_plan.plan

    def _run_kottinger_offline(
        self,
        graph: GridGraph,
        agents: list[Agent],
        cell_spec: CellSpec,
        result: CellResult,
    ) -> None:
        """Run the Kottinger baseline and record its SELF-REPORTED metrics.

        Kottinger solves the instance itself (its own nominal LNS plan),
        injects ``--numDelays`` delay events, repairs on the constrained
        graph, and reports aggregate SoC/runtime/delay stats via CSV. We
        record those numbers directly — there is no rollout-able Plan, so
        this does NOT go through ``_run_rollouts`` (see the wrapper module
        docstring and the RQ3 brief). The simulator's per-step Bernoulli
        model and Kottinger's discrete ``numDelays`` regime are different
        delay semantics; mixing them by re-simulating a reconstructed plan
        would change the comparison, so we keep Kottinger on its own axis
        and surface its reported SoC overhead.
        """
        num_delays = int(cell_spec.delay_model_config.get("kottinger_num_delays", 1))
        time_limit = float(self.load_config().get("kottinger_time_limit_s", 30.0))
        t0 = time.perf_counter()
        try:
            kres = KottingerDelayBaseline().solve_offline(
                graph, agents, time_limit_s=time_limit, num_delays=num_delays
            )
        except SolverNotFoundError as exc:
            self.console.log(
                f"[yellow]warning[/yellow]: kottinger binary missing ({exc}); "
                f"recording status=binary_missing"
            )
            result.status = "binary_missing"
            result.error_message = f"SolverNotFoundError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return
        except SolverTimeoutError as exc:
            result.status = "timeout"
            result.error_message = f"SolverTimeoutError: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return
        except (SolverError, ValueError) as exc:
            # SolverError covers the upstream crash-before-CSV / no-success
            # row case; ValueError covers numDelays >= n_agents.
            result.status = "cert_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return
        result.certifier_runtime_s = time.perf_counter() - t0

        # Record Kottinger's self-reported metrics. We map its constrained-
        # graph repair onto the RQ3 SoC fields so the analysis layer can
        # report SoC overhead on the SAME axes as the other baselines:
        #   nominal_soc        <- Kottinger's own initial-soc
        #   executed_soc_mean  <- Kottinger's repaired LNS-SIPP-CG soc
        # (deterministic single value, not a Monte-Carlo mean). The CBS-CG
        # number and raw fields are kept in diagnostics for completeness.
        result.diagnostics["mode"] = "kottinger_offline"
        result.diagnostics["solver_used"] = kres.solver_used
        result.diagnostics["kottinger_num_delays"] = num_delays
        result.diagnostics["kottinger_example_id"] = kres.example_id
        result.diagnostics["kottinger_num_agents"] = kres.num_agents
        result.diagnostics["kottinger_initial_soc"] = kres.initial_soc
        result.diagnostics["kottinger_initial_runtime_s"] = kres.initial_runtime_s
        result.diagnostics["kottinger_replan_lns_sipp_cg_soc"] = kres.replan_lns_sipp_cg_soc
        result.diagnostics["kottinger_replan_lns_sipp_cg_runtime_s"] = (
            kres.replan_lns_sipp_cg_runtime_s
        )
        result.diagnostics["kottinger_replan_lns_sipp_cg_num_delays"] = (
            kres.replan_lns_sipp_cg_num_delays
        )
        result.diagnostics["kottinger_replan_cbs_cg_soc"] = kres.replan_cbs_cg_soc
        result.diagnostics["kottinger_replan_cbs_cg_runtime_s"] = kres.replan_cbs_cg_runtime_s
        result.diagnostics["kottinger_replan_cbs_cg_num_delays"] = kres.replan_cbs_cg_num_delays

        if kres.initial_soc is not None:
            result.nominal_soc = float(kres.initial_soc)
        # Prefer the LNS-SIPP-CG repair (Kottinger's headline constrained-
        # graph result); fall back to CBS-CG if LNS-SIPP-CG was unreported.
        repaired = kres.replan_lns_sipp_cg_soc
        if repaired is None:
            repaired = kres.replan_cbs_cg_soc
        if repaired is not None:
            result.executed_soc_mean = float(repaired)
        # Kottinger's offline constrained-graph repair is delay-robust by
        # construction (it repairs against the injected delays), so we mark
        # the cell ok with a deterministic success_rate of 1.0; the SoC
        # overhead is the substantive comparison signal.
        result.success_rate = 1.0
        result.status = "ok"

    def _run_rollouts(
        self,
        plan: Plan,
        p_d: float,
        cell_spec: CellSpec,
        result: CellResult,
    ) -> None:
        model = BernoulliDelayModel(p_d=p_d)
        seed = derive_seed(cell_spec.scen_seed, cell_spec.method)
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
