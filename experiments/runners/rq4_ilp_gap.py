"""RQ4 — optimality-gap comparison: greedy SlackCertify vs. ILP exact.

This runner is structurally different from RQ1/2/3/5: only one
comparison axis (greedy vs. ILP) and capped at small ``n`` because the
wait-insertion ILP scales poorly. Each ``(map, n, scen_seed)`` cell
emits one row per method, and the slack_certify_bounded row carries a
post-hoc ``gap_pct`` / ``ilp_waits`` / ``matched_optimum`` diagnostic
when both methods finished cleanly.

The two methods share a wait-feasibility precheck (corridor-swap
case): when one side reports ``wait_infeasible`` and the other does
not, that is a precheck-disagreement bug worth surfacing. The runner
logs a warning per such cell — without failing the cell — so the
Phase 8 analysis can pick it up.
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
from slackcertify.ilp.solver import ILPResult, solve_optimal_wait_insertion
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.simulate.rollout import monte_carlo_rollout
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import SolverError, SolverNotFoundError
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.utils.seed import derive_seed

__all__ = ["RQ4ILPGapRunner"]


class RQ4ILPGapRunner(ExperimentRunner):
    """RQ4 runner — Slack-Certify greedy waits vs. ILP exact optimum."""

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

    def run_all(self) -> None:
        """Override to group per-method cells and compute the optimality gap.

        Cells are grouped by ``(map, n_agents, scen_seed)`` so both
        methods for a given instance run together, allowing
        ``gap_pct`` / ``matched_optimum`` to be attached to the
        slack_certify_bounded row before either row is written.

        Resume semantics are preserved: cells already recorded in the
        manifest are skipped per-method.
        """
        completed = self.resume_from_manifest()
        groups: dict[tuple[str, int, int], list[CellSpec]] = {}
        group_order: list[tuple[str, int, int]] = []
        for cell in self.enumerate_cells():
            key = (cell.map_name, cell.n_agents, cell.scen_seed)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(cell)

        for group_key in group_order:
            cells = groups[group_key]
            pending = [c for c in cells if c.key() not in completed]
            if not pending:
                for c in cells:
                    self.console.log(f"skip {c.key()} (already in manifest)")
                continue

            results: dict[str, CellResult] = {}
            for cell in pending:
                try:
                    result = self.run_cell(cell)
                except Exception as exc:  # noqa: BLE001
                    import traceback

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
                results[cell.method] = result

            self._attach_gap(results, group_key)
            self._check_precheck_consistency(results, group_key)

            # Emit in the order they were enumerated so the CSV row order
            # is stable across runs.
            for cell in pending:
                if cell.method in results:
                    self.append_result(results[cell.method])
                    self.console.log(
                        f"cell {cell.key()}: status={results[cell.method].status} "
                        f"success_rate={results[cell.method].success_rate:.4f}"
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

        plan_to_test = self._apply_method(nominal_plan, agents, cell_spec, cfg, delta, result)
        if plan_to_test is None:
            return result

        self._run_rollouts(plan_to_test, cell_spec, delta, result)
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
        delta: int,
        result: CellResult,
    ) -> Plan | None:
        method = cell_spec.method
        t0 = time.perf_counter()
        if method == "slack_certify_bounded":
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
        if method == "ilp_exact":
            return self._apply_ilp(nominal_plan, cfg, delta, result, t0)
        result.status = "cert_failed"
        result.error_message = f"unknown method {method!r}"
        return None

    def _apply_ilp(
        self,
        nominal_plan: Plan,
        cfg: dict[str, Any],
        delta: int,
        result: CellResult,
        t0: float,
    ) -> Plan | None:
        ilp_time_limit = float(cfg.get("ilp_time_limit_s", 3600.0))
        try:
            ilp: ILPResult = solve_optimal_wait_insertion(
                nominal_plan, delta=delta, time_limit_s=ilp_time_limit
            )
        except Exception as exc:  # noqa: BLE001
            result.status = "rollout_failed"
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.certifier_runtime_s = time.perf_counter() - t0
            return None

        result.certifier_runtime_s = time.perf_counter() - t0
        result.diagnostics["mode"] = "ilp_exact"
        result.diagnostics["delta"] = delta
        result.diagnostics["ilp_status"] = ilp.status
        result.diagnostics["ilp_runtime_s"] = ilp.runtime_s
        result.diagnostics["ilp_solver_name"] = ilp.solver_name
        if ilp.diagnostics:
            result.diagnostics["ilp_diagnostics"] = ilp.diagnostics

        if ilp.status == "optimal":
            assert ilp.certified_plan is not None
            assert ilp.objective_value is not None
            result.total_waits = int(ilp.objective_value)
            result.status = "ok"
            return ilp.certified_plan
        if ilp.status == "feasible":
            assert ilp.certified_plan is not None
            assert ilp.objective_value is not None
            result.total_waits = int(ilp.objective_value)
            result.diagnostics["ilp_gap"] = ilp.gap
            result.status = "ok"
            return ilp.certified_plan
        if ilp.status == "timeout":
            result.status = "timeout"
            result.error_message = (
                f"ILP timed out after {ilp.runtime_s:.2f}s with no feasible solution"
            )
            return None
        if ilp.status == "infeasible":
            if ilp.diagnostics.get("reason") == "wait_infeasible":
                result.status = "wait_infeasible"
                result.error_message = "ILP precheck: wait-infeasible"
            else:
                result.status = "cert_failed"
                result.error_message = (
                    f"ILP infeasible (raw_status="
                    f"{ilp.diagnostics.get('raw_status', 'unknown')})"
                )
            return None
        # ilp.status == "error"
        result.status = "rollout_failed"
        result.error_message = (
            f"ILP error (raw_status=" f"{ilp.diagnostics.get('raw_status', 'unknown')})"
        )
        return None

    def _run_rollouts(
        self,
        plan: Plan,
        cell_spec: CellSpec,
        delta: int,
        result: CellResult,
    ) -> None:
        model = BoundedDelayModel(delta=delta)
        seed = derive_seed(cell_spec.scen_seed, f"{cell_spec.method}/delta={delta}")
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
        # If status was already set (e.g. "feasible" mapped to "ok"
        # earlier), preserve it; otherwise mark "ok".
        if result.status not in ("ok",):
            result.status = "ok"

    def _attach_gap(
        self,
        results: dict[str, CellResult],
        group_key: tuple[str, int, int],
    ) -> None:
        sb = results.get("slack_certify_bounded")
        ilp = results.get("ilp_exact")
        if sb is None or ilp is None:
            return
        if sb.status != "ok" or ilp.status != "ok":
            return
        slack_waits = int(sb.total_waits)
        ilp_waits = int(ilp.total_waits)
        if slack_waits < 0 or ilp_waits < 0:
            return

        # Branch on whether the ILP proved optimality. The status="ok"
        # path covers both "optimal" and "feasible" (timed-out) ILP
        # outcomes, so we must label the gap accordingly: a gap against
        # an unproven upper bound is not the same statistic as a gap
        # against the true optimum.
        gap_pct = (slack_waits - ilp_waits) / max(1, ilp_waits) * 100.0
        ilp_status = ilp.diagnostics.get("ilp_status")
        sb.diagnostics["ilp_waits"] = ilp_waits

        if ilp_status == "optimal":
            sb.diagnostics["gap_pct_to_optimum"] = gap_pct
            sb.diagnostics["gap_label"] = "to_optimum"
            sb.diagnostics["ilp_proven_optimal"] = True
            sb.diagnostics["matched_optimum"] = slack_waits == ilp_waits
        elif ilp_status == "feasible":
            sb.diagnostics["gap_pct_to_feasible"] = gap_pct
            sb.diagnostics["gap_label"] = "to_feasible_upper_bound"
            sb.diagnostics["ilp_proven_optimal"] = False
            # matched_optimum is ambiguous when the ILP didn't prove
            # optimality: even an exact wait-count match could be a
            # mutual upper bound rather than the true optimum.
            sb.diagnostics["matched_optimum"] = None
            sb.diagnostics["note"] = (
                "ILP did not prove optimality; gap is to the best "
                "feasible solution found, which is an upper bound on "
                "the true optimum."
            )
        else:
            # status="ok" without an ilp_status diagnostic would be a
            # bug in _apply_ilp; skip the gap rather than emit a
            # malformed label.
            return

        if slack_waits < ilp_waits:
            # Greedy below the ILP's wait count. Against a proven
            # optimum this is a formulation-disagreement bug; against a
            # feasible upper bound it is merely informative.
            severity = "red" if ilp_status == "optimal" else "yellow"
            self.console.log(
                f"[{severity}]warning[/{severity}]: greedy waits "
                f"({slack_waits}) below ILP {ilp_status} waits "
                f"({ilp_waits}) for cell {group_key}"
            )

    def _check_precheck_consistency(
        self,
        results: dict[str, CellResult],
        group_key: tuple[str, int, int],
    ) -> None:
        sb = results.get("slack_certify_bounded")
        ilp = results.get("ilp_exact")
        if sb is None or ilp is None:
            return
        # The disagreement we care about: one side declares
        # wait_infeasible while the other does not (and didn't fail for
        # a clearly unrelated reason).
        terminal = {"ok", "wait_infeasible"}
        if sb.status not in terminal or ilp.status not in terminal:
            return
        if sb.status == ilp.status:
            return
        self.console.log(
            f"[red]warning[/red]: precheck disagreement for cell "
            f"{group_key}: slack_certify_bounded={sb.status!r} vs "
            f"ilp_exact={ilp.status!r}"
        )
        for r in (sb, ilp):
            r.diagnostics["precheck_disagreement"] = True
            r.diagnostics["peer_status"] = ilp.status if r is sb else sb.status
