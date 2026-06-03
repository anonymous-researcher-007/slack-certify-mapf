"""Unit tests for slackcertify.repair.propagation."""

from __future__ import annotations

import pytest

from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.repair.propagation import propagate_shift


def _single_agent_plan() -> Plan:
    return Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(3, 0))],
        paths=[Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0)])],
    )


@pytest.mark.unit
def test_zero_w_returns_input_plan_unchanged() -> None:
    plan = _single_agent_plan()
    result = propagate_shift(plan, 0, (2, 0), 0)
    assert result is plan


@pytest.mark.unit
def test_negative_w_rejected() -> None:
    plan = _single_agent_plan()
    with pytest.raises(ValueError, match="non-negative"):
        propagate_shift(plan, 0, (2, 0), -1)


@pytest.mark.unit
def test_unknown_agent_rejected() -> None:
    plan = _single_agent_plan()
    with pytest.raises(ValueError, match="unknown agent_id"):
        propagate_shift(plan, agent_id=99, before_vertex=(2, 0), w=1)


@pytest.mark.unit
def test_missing_vertex_rejected() -> None:
    plan = _single_agent_plan()
    with pytest.raises(ValueError, match="never visits"):
        propagate_shift(plan, agent_id=0, before_vertex=(7, 7), w=1)


@pytest.mark.unit
def test_inserts_waits_at_previous_cell() -> None:
    plan = _single_agent_plan()
    shifted = propagate_shift(plan, 0, (2, 0), 2)
    assert shifted.paths[0].vertices == [(0, 0), (1, 0), (1, 0), (1, 0), (2, 0), (3, 0)]


@pytest.mark.unit
def test_preserves_start_and_goal_endpoints() -> None:
    plan = _single_agent_plan()
    shifted = propagate_shift(plan, 0, (1, 0), 3)
    new_path = shifted.paths[0]
    agent = shifted.agents[0]
    assert new_path.vertices[0] == agent.start
    assert new_path.vertices[-1] == agent.goal


@pytest.mark.unit
def test_makespan_grows_by_w() -> None:
    plan = _single_agent_plan()
    shifted = propagate_shift(plan, 0, (2, 0), 4)
    assert shifted.paths[0].makespan == plan.paths[0].makespan + 4


@pytest.mark.unit
def test_other_agents_untouched() -> None:
    plan = Plan.from_paths(
        agents=[
            Agent(id=0, start=(0, 0), goal=(2, 0)),
            Agent(id=1, start=(0, 1), goal=(2, 1)),
        ],
        paths=[
            Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
            Path(agent_id=1, vertices=[(0, 1), (1, 1), (2, 1)]),
        ],
    )
    shifted = propagate_shift(plan, 0, (1, 0), 2)
    assert shifted.paths[1].vertices == [(0, 1), (1, 1), (2, 1)]


@pytest.mark.unit
def test_insert_at_start_prepends_waits() -> None:
    plan = Plan.from_paths(
        agents=[Agent(id=0, start=(1, 0), goal=(0, 0))],
        paths=[Path(agent_id=0, vertices=[(1, 0), (0, 0)])],
    )
    shifted = propagate_shift(plan, 0, (1, 0), 2)
    # before_vertex == start cell -> waits prepended at start.
    assert shifted.paths[0].vertices == [(1, 0), (1, 0), (1, 0), (0, 0)]


@pytest.mark.unit
def test_does_not_mutate_input_plan() -> None:
    plan = _single_agent_plan()
    original_vertices = list(plan.paths[0].vertices)
    _ = propagate_shift(plan, 0, (2, 0), 5)
    assert plan.paths[0].vertices == original_vertices
