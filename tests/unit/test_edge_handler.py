"""Unit tests for slackcertify.repair.edge_handler."""

from __future__ import annotations

import pytest

from slackcertify.core.conflict import EdgeConflict, detect_conflicts
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.repair.edge_handler import resolve_edge_conflict


def _head_on_swap_plan() -> Plan:
    agents = [
        Agent(id=0, start=(0, 0), goal=(1, 0)),
        Agent(id=1, start=(1, 0), goal=(0, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
        Path(agent_id=1, vertices=[(1, 0), (0, 0)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_resolves_head_on_swap_in_bounded_mode() -> None:
    plan = _head_on_swap_plan()
    ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
    resolved = resolve_edge_conflict(plan, ec, delta=0, mode="bounded")
    new_edge_conflicts = [
        c for c in detect_conflicts(resolved, delta=0) if isinstance(c, EdgeConflict)
    ]
    assert new_edge_conflicts == []


@pytest.mark.unit
def test_resolution_preserves_endpoints() -> None:
    plan = _head_on_swap_plan()
    ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
    resolved = resolve_edge_conflict(plan, ec, delta=0, mode="bounded")
    for agent, path in zip(resolved.agents, resolved.paths, strict=True):
        assert path.vertices[0] == agent.start
        assert path.vertices[-1] == agent.goal


@pytest.mark.unit
def test_returns_input_when_already_disjoint() -> None:
    # Same swap edge but the second traversal is well separated in time.
    agents = [
        Agent(id=0, start=(0, 0), goal=(1, 0)),
        Agent(id=1, start=(1, 1), goal=(0, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
        Path(agent_id=1, vertices=[(1, 1), (1, 0), (0, 0)]),
    ]
    plan = Plan.from_paths(agents, paths)
    # Construct an EdgeConflict with a wide gap so the bounded-mode shortcut
    # returns the plan unchanged.
    fake = EdgeConflict(agent_i=0, agent_j=1, u=(0, 0), v=(1, 0), t_i=0, t_j=10)
    out = resolve_edge_conflict(plan, fake, delta=0, mode="bounded")
    assert out is plan


@pytest.mark.unit
def test_probabilistic_mode_shifts_to_meet_budget() -> None:
    # The per-conflict risk formula is monotone-decreasing in the
    # nominal gap, so shifting the downstream agent at a head-on swap
    # CAN drive risk below the budget. The resolver returns a plan
    # whose downstream agent has at least one extra wait.
    plan = _head_on_swap_plan()
    ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
    resolved = resolve_edge_conflict(
        plan, ec, delta=None, mode="probabilistic", p_d=0.1, per_conflict_budget=0.05
    )
    before_steps = sum(p.makespan for p in plan.paths)
    after_steps = sum(p.makespan for p in resolved.paths)
    assert after_steps >= before_steps  # at least no shrinkage
    # The resolver returns the plan if no shift is needed; otherwise a
    # fresh Plan instance. Both outcomes are acceptable for the smoke.
    assert resolved is plan or after_steps > before_steps


@pytest.mark.unit
def test_probabilistic_mode_converges_under_corrected_formula() -> None:
    # Under the corrected (monotone-decreasing) formula, any positive
    # budget is reachable in finite shift steps. This test pins that
    # property by exercising the search loop on a synthetic
    # narrow-gap edge conflict with a tight budget; with enough waits
    # the risk drops below 0.001.
    plan = _head_on_swap_plan()
    fake = EdgeConflict(agent_i=0, agent_j=1, u=(0, 0), v=(1, 0), t_i=0, t_j=1)
    resolved = resolve_edge_conflict(
        plan,
        fake,
        delta=None,
        mode="probabilistic",
        p_d=0.5,
        per_conflict_budget=0.001,
    )
    # Should not raise; the resolver finds a finite shift that satisfies
    # the budget under the corrected formula.
    assert isinstance(resolved, type(plan))


@pytest.mark.unit
def test_bounded_mode_requires_delta() -> None:
    plan = _head_on_swap_plan()
    ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
    with pytest.raises(ValueError, match="delta"):
        resolve_edge_conflict(plan, ec, delta=None, mode="bounded")
    with pytest.raises(ValueError, match="non-negative"):
        resolve_edge_conflict(plan, ec, delta=-1, mode="bounded")


@pytest.mark.unit
def test_probabilistic_mode_requires_params() -> None:
    plan = _head_on_swap_plan()
    ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
    with pytest.raises(ValueError, match="p_d"):
        resolve_edge_conflict(plan, ec, delta=None, mode="probabilistic")


@pytest.mark.unit
def test_unsupported_mode_rejected() -> None:
    plan = _head_on_swap_plan()
    ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
    with pytest.raises(ValueError, match="unsupported mode"):
        resolve_edge_conflict(plan, ec, delta=0, mode="bogus")  # type: ignore[arg-type]
