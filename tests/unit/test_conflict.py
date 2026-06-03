"""Unit tests for slackcertify.core.conflict."""

from __future__ import annotations

import pytest

from slackcertify.core.conflict import (
    EdgeConflict,
    VertexConflict,
    detect_conflicts,
)
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.core.validators import assert_conflict_free


@pytest.fixture
def grid() -> GridGraph:
    return GridGraph(4, 4, set())


def _two_agent_shared_vertex_plan(grid: GridGraph) -> Plan:
    """Two agents whose paths cross at (1, 1) at t = 2."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 1)),
        Agent(id=1, start=(1, 3), goal=(1, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 1), (2, 1)]),
        Path(agent_id=1, vertices=[(1, 3), (1, 2), (1, 1), (1, 0)]),
    ]
    plan = Plan.from_paths(agents, paths)
    # Plan is structurally well-formed against the grid (no diagonal moves).
    from slackcertify.core.validators import validate_plan

    validate_plan(plan, grid)
    return plan


@pytest.mark.unit
def test_vertex_conflict_at_shared_cell(grid: GridGraph) -> None:
    plan = _two_agent_shared_vertex_plan(grid)
    conflicts = detect_conflicts(plan, delta=0)
    vc = [c for c in conflicts if isinstance(c, VertexConflict)]
    assert VertexConflict(agent_i=0, agent_j=1, vertex=(1, 1), t_i=2, t_j=2) in vc


@pytest.mark.unit
def test_no_edge_conflict_in_shared_vertex_plan(grid: GridGraph) -> None:
    plan = _two_agent_shared_vertex_plan(grid)
    ec = [c for c in detect_conflicts(plan, delta=0) if isinstance(c, EdgeConflict)]
    assert ec == []


@pytest.mark.unit
def test_edge_conflict_head_on_swap() -> None:
    agents = [
        Agent(id=0, start=(0, 0), goal=(1, 0)),
        Agent(id=1, start=(1, 0), goal=(0, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
        Path(agent_id=1, vertices=[(1, 0), (0, 0)]),
    ]
    plan = Plan.from_paths(agents, paths)
    conflicts = detect_conflicts(plan, delta=0)
    assert any(isinstance(c, EdgeConflict) for c in conflicts)


@pytest.mark.unit
def test_delta_widens_window(grid: GridGraph) -> None:
    # Two agents visit (2, 0) at t=1 (agent 0) and t=3 (agent 1).
    # delta=0 -> no vertex conflict at (2, 0); delta=2 -> one shows up.
    agents = [
        Agent(id=0, start=(1, 0), goal=(3, 0)),
        Agent(id=1, start=(2, 3), goal=(2, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(1, 0), (2, 0), (3, 0)]),
        Path(agent_id=1, vertices=[(2, 3), (2, 2), (2, 1), (2, 0)]),
    ]
    plan = Plan.from_paths(agents, paths)

    narrow = [c for c in detect_conflicts(plan, delta=0) if isinstance(c, VertexConflict)]
    assert all(c.vertex != (2, 0) for c in narrow)

    wide = [c for c in detect_conflicts(plan, delta=2) if isinstance(c, VertexConflict)]
    assert any(c.vertex == (2, 0) and c.t_i == 1 and c.t_j == 3 for c in wide)


@pytest.mark.unit
def test_assert_conflict_free_raises(grid: GridGraph) -> None:
    plan = _two_agent_shared_vertex_plan(grid)
    with pytest.raises(ValueError, match="conflict"):
        assert_conflict_free(plan, delta=0)


@pytest.mark.unit
def test_negative_delta_rejected(grid: GridGraph) -> None:
    plan = _two_agent_shared_vertex_plan(grid)
    with pytest.raises(ValueError, match="non-negative"):
        detect_conflicts(plan, delta=-1)
