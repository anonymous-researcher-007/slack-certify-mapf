"""Unit tests for slackcertify.core.plan."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.core.validators import validate_plan


@pytest.fixture
def grid() -> GridGraph:
    return GridGraph(4, 4, set())


@pytest.mark.unit
def test_agent_minimal() -> None:
    a = Agent(id=0, start=(0, 0), goal=(3, 3))
    assert a.id == 0
    assert a.start == (0, 0)
    assert a.goal == (3, 3)


@pytest.mark.unit
def test_path_basic_properties() -> None:
    p = Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 0)])
    assert len(p) == 3
    assert p.makespan == 2
    # Arrival time = 1 because the agent is at (1, 0) from t=1 onward.
    assert p.arrival_time == 1


@pytest.mark.unit
def test_path_endpoint_validation() -> None:
    a = Agent(id=0, start=(0, 0), goal=(2, 0))
    # Wrong start.
    with pytest.raises(ValidationError, match="starts at"):
        Path.model_validate(
            {"agent_id": 0, "vertices": [(1, 0), (2, 0)]},
            context={"agent": a},
        )
    # Wrong goal.
    with pytest.raises(ValidationError, match="ends at"):
        Path.model_validate(
            {"agent_id": 0, "vertices": [(0, 0), (1, 0)]},
            context={"agent": a},
        )


@pytest.mark.unit
def test_validator_allows_goal_revisit(grid: GridGraph) -> None:
    # Convention B: the agent may reach its goal, step off, and return,
    # as long as it ENDS at the goal. (Was rejected under convention A.)
    a = Agent(id=0, start=(0, 0), goal=(2, 0))
    Path.model_validate(
        {"agent_id": 0, "vertices": [(0, 0), (1, 0), (2, 0), (1, 0), (2, 0), (2, 0)]},
        context={"graph": grid, "agent": a},
    )


@pytest.mark.unit
def test_validator_rejects_not_ending_at_goal(grid: GridGraph) -> None:
    # Convention B still requires the FINAL cell to be the goal.
    a = Agent(id=0, start=(0, 0), goal=(2, 0))
    with pytest.raises(ValidationError, match="ends at"):
        Path.model_validate(
            {"agent_id": 0, "vertices": [(0, 0), (1, 0)]},
            context={"graph": grid, "agent": a},
        )


@pytest.mark.unit
def test_validator_still_checks_adjacency(grid: GridGraph) -> None:
    # Relaxing the goal-terminality block must not drop adjacency checks:
    # a two-cell jump (0,0)->(2,0) is non-adjacent and must still fail.
    a = Agent(id=0, start=(0, 0), goal=(2, 0))
    with pytest.raises(ValidationError, match="not adjacent"):
        Path.model_validate(
            {"agent_id": 0, "vertices": [(0, 0), (2, 0)]},
            context={"graph": grid, "agent": a},
        )


@pytest.mark.unit
def test_path_graph_adjacency(grid: GridGraph) -> None:
    # Diagonal step in a 4-connected grid should fail.
    with pytest.raises(ValidationError, match="not adjacent"):
        Path.model_validate(
            {"agent_id": 0, "vertices": [(0, 0), (1, 1)]},
            context={"graph": grid},
        )


@pytest.mark.unit
def test_plan_from_paths_computes_costs() -> None:
    a = [Agent(id=0, start=(0, 0), goal=(2, 0)), Agent(id=1, start=(0, 1), goal=(0, 1))]
    p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]), Path(agent_id=1, vertices=[(0, 1)])]
    plan = Plan.from_paths(a, p)
    assert plan.makespan == 2
    # Agent 0 contributes arrival_time=2; agent 1 stays put -> 0.
    assert plan.sum_of_costs == 2


@pytest.mark.unit
def test_plan_vertex_visit_pads_with_goal() -> None:
    a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
    p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
    plan = Plan.from_paths(a, p)
    assert plan.vertex_visit(0, 0) == (0, 0)
    assert plan.vertex_visit(0, 1) == (1, 0)
    assert plan.vertex_visit(0, 99) == (1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        plan.vertex_visit(0, -1)


@pytest.mark.unit
def test_plan_validator_rejects_unknown_agent_path() -> None:
    a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    p = [Path(agent_id=7, vertices=[(0, 0)])]  # mismatched id
    with pytest.raises(ValidationError, match="do not match"):
        Plan.from_paths(a, p)


@pytest.mark.unit
def test_validate_plan_with_graph(grid: GridGraph) -> None:
    a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
    p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
    validate_plan(Plan.from_paths(a, p), grid)
