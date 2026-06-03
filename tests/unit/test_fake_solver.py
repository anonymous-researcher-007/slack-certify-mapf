"""Unit tests for slackcertify.solvers._fake.FakeSolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath
from slackcertify.io.plan_io import save_plan
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import SolverError


@pytest.mark.unit
def test_fake_solver_with_fixture_returns_plan(tmp_path: Path) -> None:
    plan = Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(2, 0))],
        paths=[PlanPath(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])],
    )
    fixture = tmp_path / "fixture.json"
    save_plan(plan, fixture)

    grid = GridGraph(3, 3, set())
    agents = [Agent(id=0, start=(0, 0), goal=(2, 0))]

    result = FakeSolver().solve(grid, agents, time_limit_s=0.0, fixture_path=fixture)
    assert result == plan


@pytest.mark.unit
def test_fake_solver_with_default_fixture(tmp_path: Path) -> None:
    plan = Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(0, 0))],
        paths=[PlanPath(agent_id=0, vertices=[(0, 0)])],
    )
    fixture = tmp_path / "fixture.json"
    save_plan(plan, fixture)
    solver = FakeSolver(fixture_path=fixture)
    grid = GridGraph(3, 3, set())
    agents = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    assert solver.solve(grid, agents, time_limit_s=0.0) == plan


@pytest.mark.unit
def test_fake_solver_without_fixture_uses_bfs() -> None:
    grid = GridGraph(4, 4, set())
    agents = [
        Agent(id=0, start=(0, 0), goal=(3, 0)),
        Agent(id=1, start=(0, 3), goal=(3, 3)),
    ]
    plan = FakeSolver().solve(grid, agents, time_limit_s=0.0)
    assert {p.agent_id for p in plan.paths} == {0, 1}
    # Paths reach the goals.
    for a, p in zip(agents, plan.paths, strict=True):
        assert p.vertices[0] == a.start
        assert p.vertices[-1] == a.goal


@pytest.mark.unit
def test_fake_solver_unreachable_goal_raises() -> None:
    grid = GridGraph(3, 3, {(1, 0), (1, 1), (1, 2)})  # column 1 fully blocked
    agents = [Agent(id=0, start=(0, 0), goal=(2, 0))]
    with pytest.raises(SolverError, match="no BFS path"):
        FakeSolver().solve(grid, agents, time_limit_s=0.0)


@pytest.mark.unit
def test_fake_solver_rejects_invalid_fixture_type() -> None:
    grid = GridGraph(3, 3, set())
    agents = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    with pytest.raises(TypeError, match="str or Path"):
        FakeSolver().solve(grid, agents, time_limit_s=0.0, fixture_path=42)
