"""In-process FakeSolver — used by tests and smoke runs.

The fake solver bypasses the subprocess machinery entirely. It either
loads a pre-baked :class:`Plan` from a JSON fixture (preferred for
deterministic regression tests) or falls back to per-agent BFS shortest
paths (fast, may have inter-agent conflicts — *not* a real solve, but
adequate as a stub plan for the slack certifier to chew on).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from slackcertify.core.graph import Cell, GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath
from slackcertify.io.plan_io import load_plan
from slackcertify.solvers.base import SolverError

__all__ = ["FakeSolver"]


class FakeSolver:
    """In-process stand-in for a real MAPF solver.

    Parameters
    ----------
    fixture_path
        Optional default fixture; can be overridden per-call via the
        ``fixture_path`` keyword in :meth:`solve`.
    """

    NAME: ClassVar[str] = "fake"

    def __init__(self, fixture_path: str | Path | None = None) -> None:
        """Store an optional default fixture path used by every :meth:`solve`."""
        self._fixture: Path | None = Path(fixture_path) if fixture_path else None

    def solve(
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        time_limit_s: float,
        **kwargs: object,
    ) -> Plan:
        """Return a plan, either from a fixture or from per-agent BFS shortest paths.

        Parameters
        ----------
        graph
            Used only for the BFS fallback; ignored when a fixture path
            is supplied.
        agents
            Agent list; must match the fixture if one is provided.
        time_limit_s
            Accepted for interface parity with real solvers; unused.
        fixture_path
            Per-call override for the fixture set on construction.
        """
        _ = time_limit_s  # accepted for interface parity
        fixture = kwargs.get("fixture_path")
        path: Path | None
        if isinstance(fixture, (str, Path)):
            path = Path(fixture)
        elif fixture is None:
            path = self._fixture
        else:
            raise TypeError(f"fixture_path must be a str or Path, got {type(fixture).__name__}")

        if path is not None:
            return load_plan(path)
        return _bfs_independent_plan(graph, list(agents))


def _bfs_independent_plan(graph: GridGraph, agents: list[Agent]) -> Plan:
    """Return a Plan whose per-agent paths are independent BFS shortest paths."""
    plan_paths: list[PlanPath] = []
    for agent in agents:
        verts = _bfs_shortest(graph, agent.start, agent.goal)
        if verts is None:
            raise SolverError(
                f"FakeSolver: no BFS path for agent {agent.id} "
                f"from {agent.start} to {agent.goal}"
            )
        plan_paths.append(PlanPath(agent_id=agent.id, vertices=verts))
    return Plan.from_paths(agents, plan_paths)


def _bfs_shortest(graph: GridGraph, start: Cell, goal: Cell) -> list[Cell] | None:
    """Return the BFS shortest path from ``start`` to ``goal`` (or ``None``)."""
    if start == goal:
        return [start]
    if not graph.is_free(start) or not graph.is_free(goal):
        return None
    visited: set[Cell] = {start}
    parent: dict[Cell, Cell] = {}
    queue: deque[Cell] = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in graph.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            parent[nbr] = node
            if nbr == goal:
                # Reconstruct.
                path: list[Cell] = [nbr]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nbr)
    return None
