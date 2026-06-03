"""Solve the wait-insertion ILP and re-construct the certified plan."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    import pulp
except ImportError as exc:  # pragma: no cover - exercised by users without [ilp]
    raise ImportError(
        "ILP support requires pulp. Install with: pip install 'slackcertify[ilp]'"
    ) from exc

from slackcertify.core.conflict import detect_conflicts
from slackcertify.core.plan import Path as PlanPath
from slackcertify.core.plan import Plan
from slackcertify.ilp._solver_detection import available_solvers
from slackcertify.ilp.encoding import build_wait_insertion_ilp
from slackcertify.repair.wait_feasibility import is_wait_resolvable

__all__ = ["ILPResult", "solve_optimal_wait_insertion"]


SolverName = Literal["highs", "cbc", "gurobi"]
ResultStatus = Literal["optimal", "feasible", "infeasible", "timeout", "error"]


@dataclass
class ILPResult:
    """Outcome of a single :func:`solve_optimal_wait_insertion` call."""

    certified_plan: Plan | None
    objective_value: int | None
    status: ResultStatus
    gap: float
    runtime_s: float
    solver_name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def solve_optimal_wait_insertion(
    plan: Plan,
    delta: int,
    time_limit_s: float = 3600.0,
    solver: SolverName = "highs",
    msg: bool = False,
) -> ILPResult:
    """Solve the optimal wait-insertion ILP for ``(plan, delta)``.

    Selects an MIP backend (defaulting to HiGHS), passes the wall-clock
    time limit to the solver, reads the integer wait values back into
    a fully-realised :class:`Plan`, and returns an :class:`ILPResult`
    summarising the optimisation outcome.

    Parameters
    ----------
    plan
        Nominal plan whose path topology is preserved.
    delta
        Δ-disjointness tolerance.
    time_limit_s
        Wall-clock budget passed to the backend.
    solver
        ``"highs"``, ``"cbc"`` or ``"gurobi"``. Falls back to the
        first :func:`available_solvers` entry if the requested backend
        is missing; raises :class:`RuntimeError` if no backend is
        available at all.
    msg
        If ``True``, the solver prints its progress to stdout.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> r = solve_optimal_wait_insertion(Plan.from_paths(a, p), delta=0)
    >>> r.status, r.objective_value
    ('optimal', 0)
    """
    t_start = time.perf_counter()
    chosen = _resolve_solver(solver)

    # Wait-feasibility precheck: a head-on swap on a topology where
    # both agents traverse exactly the same cell set has no
    # wait-insertion solution. Return early with a clean
    # "infeasible" status instead of letting the LP silently accept
    # a model that doesn't constrain the induced post-shift conflicts.
    initial_conflicts = detect_conflicts(plan, delta=delta)
    feasible, unresolvable = is_wait_resolvable(plan, initial_conflicts)
    if not feasible:
        return ILPResult(
            certified_plan=None,
            objective_value=None,
            status="infeasible",
            gap=0.0,
            runtime_s=time.perf_counter() - t_start,
            solver_name="precheck",
            diagnostics={
                "reason": "wait_infeasible",
                "unresolvable_conflicts": [repr(c) for c in unresolvable],
            },
        )

    encoding = build_wait_insertion_ilp(plan, delta=delta, conflicts=initial_conflicts)
    backend = _make_backend(chosen, time_limit_s=time_limit_s, msg=msg)

    t0 = time.perf_counter()
    encoding.problem.solve(backend)
    runtime = time.perf_counter() - t0

    raw_status = pulp.LpStatus[encoding.problem.status]
    diagnostics: dict[str, Any] = {
        "raw_status": raw_status,
        "elapsed_s": runtime,
        "n_constraints": len(encoding.problem.constraints),
        "n_conflicts": len(encoding.conflicts),
    }

    if encoding.problem.status == pulp.LpStatusOptimal:
        certified = _augment_plan_with_waits(plan, encoding.wait_vars)
        objective = _objective_int(encoding.problem)
        return ILPResult(
            certified_plan=certified,
            objective_value=objective,
            status="optimal",
            gap=0.0,
            runtime_s=runtime,
            solver_name=chosen,
            diagnostics=diagnostics,
        )

    if encoding.problem.status == pulp.LpStatusInfeasible:
        return ILPResult(
            certified_plan=None,
            objective_value=None,
            status="infeasible",
            gap=math.inf,
            runtime_s=runtime,
            solver_name=chosen,
            diagnostics=diagnostics,
        )

    objective_val = pulp.value(encoding.problem.objective)
    has_feasible = objective_val is not None and not math.isnan(float(objective_val))

    if encoding.problem.status == pulp.LpStatusNotSolved or runtime >= time_limit_s * 0.9:
        status: ResultStatus = "timeout" if runtime >= time_limit_s * 0.5 else "error"
        certified_to: Plan | None = (
            _augment_plan_with_waits(plan, encoding.wait_vars) if has_feasible else None
        )
        return ILPResult(
            certified_plan=certified_to,
            objective_value=int(round(float(objective_val))) if has_feasible else None,
            status=status,
            gap=math.nan,
            runtime_s=runtime,
            solver_name=chosen,
            diagnostics={**diagnostics, "best_bound_known": has_feasible},
        )

    if has_feasible:
        return ILPResult(
            certified_plan=_augment_plan_with_waits(plan, encoding.wait_vars),
            objective_value=int(round(float(objective_val))),
            status="feasible",
            gap=math.nan,
            runtime_s=runtime,
            solver_name=chosen,
            diagnostics=diagnostics,
        )

    return ILPResult(
        certified_plan=None,
        objective_value=None,
        status="error",
        gap=math.nan,
        runtime_s=runtime,
        solver_name=chosen,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------- internals


def _resolve_solver(requested: str) -> str:
    """Return ``requested`` if available, else the first available backend."""
    av = available_solvers()
    if not av:
        raise RuntimeError("no ILP solver available; install with: pip install 'slackcertify[ilp]'")
    if requested in av:
        return requested
    return av[0]


def _make_backend(name: str, *, time_limit_s: float, msg: bool) -> pulp.LpSolver:
    """Instantiate the requested MIP backend with the given wall-clock budget."""
    if name == "highs":
        if hasattr(pulp, "HiGHS"):
            try:
                return pulp.HiGHS(timeLimit=time_limit_s, msg=msg)
            except Exception:  # noqa: BLE001 - fall back if Python API blows up
                pass
        if hasattr(pulp, "HiGHS_CMD"):
            return pulp.HiGHS_CMD(timeLimit=time_limit_s, msg=msg)
        raise RuntimeError("PuLP build does not expose HiGHS")
    if name == "cbc":
        return pulp.PULP_CBC_CMD(timeLimit=time_limit_s, msg=msg)
    if name == "gurobi":
        return pulp.GUROBI_CMD(timeLimit=time_limit_s, msg=msg)
    raise ValueError(f"unknown solver backend: {name!r}")


def _objective_int(problem: pulp.LpProblem) -> int:
    """Return the integer-rounded objective value of a solved LP problem."""
    val = pulp.value(problem.objective)
    if val is None:
        return 0
    return int(round(float(val)))


def _augment_plan_with_waits(plan: Plan, wait_vars: dict[tuple[int, int], pulp.LpVariable]) -> Plan:
    """Insert ``round(w_{i,k})`` extra copies of ``path[k]`` after every visit."""
    new_paths: list[PlanPath] = []
    for path in plan.paths:
        verts = list(path.vertices)
        out: list[tuple[int, int]] = []
        for k, v in enumerate(verts):
            out.append(v)
            raw = pulp.value(wait_vars[(path.agent_id, k)])
            w = int(round(float(raw))) if raw is not None else 0
            out.extend([v] * max(w, 0))
        new_paths.append(PlanPath(agent_id=path.agent_id, vertices=out))
    return Plan.from_paths(plan.agents, new_paths)
