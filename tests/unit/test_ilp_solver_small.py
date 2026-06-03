"""Unit tests for slackcertify.ilp.solver on small fixtures.

Skips at module-load time when the ``[ilp]`` optional extras are
absent (no PuLP). ``solve_optimal_wait_insertion`` is imported
lazily inside each test body so the module's top of file remains
import-statement-only — there's no E402 to suppress.

``slackcertify.ilp._solver_detection`` is safe to import without
PuLP (its internals lazy-import inside ``_has_*`` helpers), so
``available_solvers`` lives at module level next to ``_AVAILABLE``.
"""

from __future__ import annotations

import pytest

from slackcertify.certify.bounded import is_delta_certified
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.ilp import _solver_detection
from slackcertify.ilp._solver_detection import available_solvers
from slackcertify.repair import slack_certify

pulp = pytest.importorskip("pulp")

_AVAILABLE = available_solvers()
_REASON = "no ILP solver installed; install slackcertify[ilp]"


def _two_agent_phase4_fixture() -> Plan:
    """The Phase-4 integration check: vertex conflict at (2, 2) at t=2."""
    agents = [
        Agent(id=0, start=(0, 2), goal=(3, 2)),
        Agent(id=1, start=(2, 0), goal=(2, 3)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 2), (1, 2), (2, 2), (3, 2)]),
        Path(agent_id=1, vertices=[(2, 0), (2, 1), (2, 2), (2, 3)]),
    ]
    return Plan.from_paths(agents, paths)


def _three_agent_kottinger_fixture() -> Plan:
    agents = [
        Agent(id=0, start=(0, 1), goal=(2, 1)),
        Agent(id=1, start=(1, 0), goal=(1, 2)),
        Agent(id=2, start=(2, 0), goal=(2, 2)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 1), (1, 1), (2, 1)]),
        Path(agent_id=1, vertices=[(1, 0), (1, 1), (1, 2)]),
        Path(agent_id=2, vertices=[(2, 0), (2, 1), (2, 1), (2, 2)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_solver_finds_known_optimum_on_2_agent_fixture() -> None:
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    plan = _two_agent_phase4_fixture()
    result = solve_optimal_wait_insertion(plan, delta=1, time_limit_s=30.0)
    assert result.status == "optimal"
    assert result.objective_value == 2
    assert result.gap == 0.0
    assert result.certified_plan is not None
    assert is_delta_certified(result.certified_plan, delta=1)


@pytest.mark.unit
@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_solver_matches_slack_certify_on_simple_instance() -> None:
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    plan = _two_agent_phase4_fixture()
    ilp_result = solve_optimal_wait_insertion(plan, delta=1, time_limit_s=30.0)
    sc_plan, sc_cert = slack_certify(plan, mode="bounded", delta=1)
    assert ilp_result.objective_value is not None
    # ILP is exact so its objective must match or beat the greedy.
    assert ilp_result.objective_value <= sc_cert.total_wait_inserted


@pytest.mark.unit
@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_solver_handles_timeout_cleanly() -> None:
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    plan = _three_agent_kottinger_fixture()
    result = solve_optimal_wait_insertion(plan, delta=1, time_limit_s=0.01)
    # Either the tiny instance still solves under 10 ms, or we get a clean
    # timeout / feasible status — anything but "error".
    assert result.status in {"optimal", "feasible", "timeout"}
    if result.status == "timeout":
        assert result.certified_plan is None or "best_bound_known" in result.diagnostics


@pytest.mark.unit
@pytest.mark.skipif("cbc" not in _AVAILABLE, reason="CBC backend not available")
def test_solver_works_with_cbc_backend() -> None:
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    plan = _two_agent_phase4_fixture()
    result = solve_optimal_wait_insertion(plan, delta=1, time_limit_s=30.0, solver="cbc")
    assert result.status == "optimal"
    assert result.solver_name == "cbc"
    assert result.objective_value == 2


@pytest.mark.unit
@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_solver_falls_back_when_requested_backend_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ask for "gurobi" which we don't have; resolver should fall back to
    # the first available backend (highs) and still produce an optimum.
    import slackcertify.ilp.solver as solver_mod
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    monkeypatch.setattr(solver_mod, "available_solvers", lambda: ["highs"])
    plan = _two_agent_phase4_fixture()
    result = solve_optimal_wait_insertion(plan, delta=1, time_limit_s=30.0, solver="gurobi")
    assert result.solver_name == "highs"
    assert result.status == "optimal"


@pytest.mark.unit
@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_ilp_rejects_corridor_swap_as_infeasible() -> None:
    """A head-on swap on a 1-D corridor must be flagged as wait-infeasible
    (either by the PART-A precheck or by the underlying LP). Pre-PART-B,
    the ILP silently returned a false 'optimal' status on this instance."""
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    n = 4
    agents = [
        Agent(id=0, start=(0, 0), goal=(0, n - 1)),
        Agent(id=1, start=(0, n - 1), goal=(0, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, i) for i in range(n)]),
        Path(agent_id=1, vertices=[(0, n - 1 - i) for i in range(n)]),
    ]
    plan = Plan.from_paths(agents, paths)

    result = solve_optimal_wait_insertion(plan, delta=1, time_limit_s=10.0)
    assert result.status == "infeasible"
    assert result.certified_plan is None
    # Either the structural precheck rejected it, or the LP itself returned
    # infeasible — both are correct outcomes for the corridor swap.
    reason = result.diagnostics.get("reason")
    if reason is None:
        # LP-side infeasibility — solver_name is the actual MIP backend.
        assert result.solver_name in {"highs", "cbc", "gurobi"}
    else:
        assert reason == "wait_infeasible"
        assert result.solver_name == "precheck"


@pytest.mark.unit
def test_solver_raises_when_no_backend_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_solver_detection, "available_solvers", lambda: [])
    # solver.py imports the function by name, so patch it there too.
    import slackcertify.ilp.solver as solver_mod
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    monkeypatch.setattr(solver_mod, "available_solvers", lambda: [])
    plan = _two_agent_phase4_fixture()
    with pytest.raises(RuntimeError, match=r"slackcertify\[ilp\]"):
        solve_optimal_wait_insertion(plan, delta=1, time_limit_s=1.0)
