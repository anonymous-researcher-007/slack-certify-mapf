"""Reservation table built from a :class:`~slackcertify.core.plan.Plan`.

A :class:`ReservationTable` materialises the (vertex → visits) and
(directed-edge → traversals) views of a plan once at construction time,
so that downstream lookups (used heavily by the certifier and the
conflict-resolution loop) are cheap.

All returned lists are sorted by ``time_step`` ascending and are *new*
list instances; callers are free to mutate them without disturbing the
table.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> agents = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> paths = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> rt = ReservationTable(Plan.from_paths(agents, paths))
>>> rt.visits_at((1, 0))
[(0, 1)]
>>> rt.edge_traversals((0, 0), (1, 0))
[(0, 0)]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from slackcertify.core.graph import Cell
from slackcertify.core.plan import Plan

__all__ = ["ReservationTable"]


@dataclass(frozen=True, slots=True)
class ReservationTable:
    """Pre-indexed view of vertex visits and directed edge traversals.

    Parameters
    ----------
    plan
        The plan whose paths should be indexed.
    """

    plan: Plan
    _vertex_index: dict[Cell, list[tuple[int, int]]]
    _edge_index: dict[tuple[Cell, Cell], list[tuple[int, int]]]

    def __init__(self, plan: Plan) -> None:
        """Pre-index every vertex and edge reservation in ``plan`` for O(1) lookup."""
        v_idx: dict[Cell, list[tuple[int, int]]] = defaultdict(list)
        e_idx: dict[tuple[Cell, Cell], list[tuple[int, int]]] = defaultdict(list)
        for path in plan.paths:
            verts = path.vertices
            for t, v in enumerate(verts):
                v_idx[v].append((path.agent_id, t))
            for t, (u, w) in enumerate(zip(verts, verts[1:], strict=False)):
                if u == w:  # waits are not edge traversals
                    continue
                e_idx[(u, w)].append((path.agent_id, t))

        # Sort once so query results come out in deterministic time order.
        for vk in v_idx:
            v_idx[vk].sort(key=lambda av: av[1])
        for ek in e_idx:
            e_idx[ek].sort(key=lambda av: av[1])

        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "_vertex_index", dict(v_idx))
        object.__setattr__(self, "_edge_index", dict(e_idx))

    def visits_at(self, vertex: Cell) -> list[tuple[int, int]]:
        """Return ``(agent_id, time_step)`` pairs that visit ``vertex``.

        Result is sorted by ``time_step`` ascending. Lookup is ``O(1)``
        (dict probe) and the returned list is a fresh copy of the
        pre-sorted index entry.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0)),
        ...      Agent(id=1, start=(0, 0), goal=(0, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0)]),
        ...      Path(agent_id=1, vertices=[(0, 0)])]
        >>> ReservationTable(Plan.from_paths(a, p)).visits_at((0, 0))
        [(0, 0), (1, 0)]
        """
        return list(self._vertex_index.get(vertex, ()))

    def edge_traversals(self, u: Cell, v: Cell) -> list[tuple[int, int]]:
        """Return ``(agent_id, t)`` pairs that traverse the directed edge ``u -> v``.

        ``t`` is the *departure* time, so the agent is at ``u`` at ``t``
        and at ``v`` at ``t + 1``. Wait actions (``u == v``) are not
        recorded.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
        >>> ReservationTable(Plan.from_paths(a, p)).edge_traversals((1, 0), (2, 0))
        [(0, 1)]
        """
        return list(self._edge_index.get((u, v), ()))

    def vertices(self) -> list[Cell]:
        """Return all vertices visited by *any* agent.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
        >>> sorted(ReservationTable(Plan.from_paths(a, p)).vertices())
        [(0, 0), (1, 0)]
        """
        return list(self._vertex_index.keys())

    def edges(self) -> list[tuple[Cell, Cell]]:
        """Return all directed edges traversed by *any* agent."""
        return list(self._edge_index.keys())
