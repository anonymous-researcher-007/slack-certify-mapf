"""Unit tests for slackcertify.solvers.base."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath
from slackcertify.solvers.base import (
    BinarySolverBase,
    SolverNotFoundError,
    SolverTimeoutError,
)


class _NoopSolver(BinarySolverBase):
    """Minimal concrete subclass: parses a fixed plan, ignores stdout."""

    NAME = "noop"

    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        return [str(self.binary_path), str(map_path), str(scen_path)]

    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        plan_paths = [PlanPath(agent_id=a.id, vertices=[a.start]) for a in agents]
        return Plan.from_paths(list(agents), plan_paths)


class _SleepSolver(_NoopSolver):
    """Concrete subclass that invokes ``/bin/sleep`` to force a timeout."""

    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        return [str(self.binary_path), "10"]


@pytest.fixture
def grid() -> GridGraph:
    return GridGraph(4, 4, set())


@pytest.fixture
def agents() -> list[Agent]:
    return [Agent(id=0, start=(0, 0), goal=(0, 0))]


@pytest.mark.unit
def test_solver_not_found_raises_with_install_hint(grid: GridGraph, agents: list[Agent]) -> None:
    solver = _NoopSolver(binary_path="/definitely/not/here")
    with pytest.raises(SolverNotFoundError, match="install_baselines"):
        solver.solve(grid, agents, time_limit_s=1.0)


@pytest.mark.unit
def test_solver_not_found_is_a_filenotfounderror(grid: GridGraph, agents: list[Agent]) -> None:
    solver = _NoopSolver(binary_path="/definitely/not/here")
    # SolverNotFoundError multiply-inherits from FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        solver.solve(grid, agents, time_limit_s=1.0)


@pytest.mark.unit
def test_subprocess_timeout_raises_solver_timeout_error(
    grid: GridGraph, agents: list[Agent]
) -> None:
    sleep_path = shutil.which("sleep")
    if sleep_path is None:
        pytest.skip("/bin/sleep not available")
    solver = _SleepSolver(binary_path=sleep_path)
    with pytest.raises(SolverTimeoutError, match="exceeded"):
        solver.solve(grid, agents, time_limit_s=0.1)


@pytest.mark.unit
def test_movingai_input_round_trips(grid: GridGraph, agents: list[Agent], tmp_path: Path) -> None:
    grid_with_obstacles = GridGraph(4, 4, {(1, 1), (2, 2)})
    solver = _NoopSolver(binary_path="/dev/null")
    map_path, scen_path = solver._write_movingai_input(  # noqa: SLF001
        grid_with_obstacles, agents, tmp_path
    )
    assert map_path.exists()
    assert scen_path.exists()
    loaded = GridGraph.from_movingai(map_path)
    assert loaded == grid_with_obstacles
    # The .scen file declares "version 1" and one bucket-0 line per agent.
    text = scen_path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "version 1"
    assert len(text) == 1 + len(agents)
