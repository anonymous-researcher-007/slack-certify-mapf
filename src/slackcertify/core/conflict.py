"""Conflict types and detection for MAPF plans under bounded delay tolerance.

A conflict is a pair of agents whose trajectories interfere within a tolerance
``delta`` (in time steps) on either a shared vertex or a reverse-traversed
edge. With ``delta == 0`` this collapses to the standard MAPF vertex / edge
conflict definitions; with ``delta > 0`` it captures the bounded-delay model
used by the slack certifier.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> agents = [Agent(id=0, start=(0, 0), goal=(1, 0)),
...           Agent(id=1, start=(1, 0), goal=(0, 0))]
>>> paths = [Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
...          Path(agent_id=1, vertices=[(1, 0), (0, 0)])]
>>> plan = Plan.from_paths(agents, paths)
>>> conflicts = detect_conflicts(plan, delta=0)
>>> any(isinstance(c, EdgeConflict) for c in conflicts)
True
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from slackcertify.core.graph import Cell
from slackcertify.core.plan import Plan

__all__ = [
    "VertexConflict",
    "EdgeConflict",
    "Conflict",
    "detect_conflicts",
    "detect_conflicts_fast",
]


@dataclass(frozen=True, slots=True)
class VertexConflict:
    """Two agents occupy the same vertex within a delta time window.

    Examples
    --------
    >>> VertexConflict(agent_i=0, agent_j=1, vertex=(2, 3), t_i=4, t_j=4)
    VertexConflict(agent_i=0, agent_j=1, vertex=(2, 3), t_i=4, t_j=4)
    """

    agent_i: int
    agent_j: int
    vertex: Cell
    t_i: int
    t_j: int


@dataclass(frozen=True, slots=True)
class EdgeConflict:
    """Two agents traverse the same edge in opposite directions within a delta window.

    Agent ``i`` moves ``u -> v`` between times ``t_i`` and ``t_i + 1``;
    agent ``j`` moves ``v -> u`` between times ``t_j`` and ``t_j + 1``.

    Examples
    --------
    >>> EdgeConflict(agent_i=0, agent_j=1, u=(0, 0), v=(1, 0), t_i=0, t_j=0)
    EdgeConflict(agent_i=0, agent_j=1, u=(0, 0), v=(1, 0), t_i=0, t_j=0)
    """

    agent_i: int
    agent_j: int
    u: Cell
    v: Cell
    t_i: int
    t_j: int


Conflict = VertexConflict | EdgeConflict


def detect_conflicts(plan: Plan, delta: int = 0) -> list[Conflict]:
    """Enumerate vertex and reverse-edge conflicts in ``plan`` under tolerance ``delta``.

    Two agents *vertex-conflict* on cell ``v`` if there exist times
    ``t_i, t_j`` with ``plan.vertex_visit(i, t_i) == v ==
    plan.vertex_visit(j, t_j)`` and ``|t_i - t_j| <= delta``. They
    *edge-conflict* on edge ``(u, v)`` if agent ``i`` traverses ``u -> v``
    starting at ``t_i`` while agent ``j`` traverses ``v -> u`` starting at
    ``t_j``, again with ``|t_i - t_j| <= delta``. All such pairs are
    returned (one entry per offending ``(agent_i, agent_j, t_i, t_j)``
    quadruple, with ``agent_i < agent_j``).

    Time horizon. The function inspects times ``0 ≤ t ≤ plan.makespan``
    for vertex conflicts and ``0 ≤ t < plan.makespan`` for edge conflicts;
    agents that have arrived at their goals continue to occupy them
    (per :meth:`Plan.vertex_visit`).

    Parameters
    ----------
    plan
        The :class:`~slackcertify.core.plan.Plan` to inspect.
    delta
        Non-negative time tolerance. ``delta=0`` is the standard MAPF
        definition.

    Returns
    -------
    list of :class:`Conflict`
        Every offending pair, in ascending lexicographic order of
        ``(agent_i, agent_j, t_i, t_j)``.

    Raises
    ------
    ValueError
        If ``delta`` is negative.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0)),
    ...      Agent(id=1, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)]),
    ...      Path(agent_id=1, vertices=[(0, 0)])]
    >>> plan = Plan.from_paths(a, p)
    >>> [c for c in detect_conflicts(plan) if isinstance(c, VertexConflict)]
    [VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=0, t_j=0)]
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    horizon = plan.makespan
    agents = sorted(a.id for a in plan.agents)
    out: list[Conflict] = []

    # Vertex conflicts: scan all (i < j, t_i, t_j) with |t_i - t_j| <= delta.
    for idx_i, i in enumerate(agents):
        for j in agents[idx_i + 1 :]:
            for t_i in range(horizon + 1):
                v_i = plan.vertex_visit(i, t_i)
                t_lo = max(0, t_i - delta)
                t_hi = min(horizon, t_i + delta)
                for t_j in range(t_lo, t_hi + 1):
                    if plan.vertex_visit(j, t_j) == v_i:
                        out.append(
                            VertexConflict(agent_i=i, agent_j=j, vertex=v_i, t_i=t_i, t_j=t_j)
                        )

    # Edge conflicts: agent i moves u->v at [t_i, t_i+1]; agent j moves
    # v->u at [t_j, t_j+1]; |t_i - t_j| <= delta.
    for idx_i, i in enumerate(agents):
        for j in agents[idx_i + 1 :]:
            for t_i in range(horizon):
                u_i = plan.vertex_visit(i, t_i)
                v_i = plan.vertex_visit(i, t_i + 1)
                if u_i == v_i:
                    continue  # i is waiting; no edge traversal
                t_lo = max(0, t_i - delta)
                t_hi = min(horizon - 1, t_i + delta)
                for t_j in range(t_lo, t_hi + 1):
                    u_j = plan.vertex_visit(j, t_j)
                    v_j = plan.vertex_visit(j, t_j + 1)
                    if u_j == v_i and v_j == u_i:
                        out.append(
                            EdgeConflict(
                                agent_i=i,
                                agent_j=j,
                                u=u_i,
                                v=v_i,
                                t_i=t_i,
                                t_j=t_j,
                            )
                        )

    out.sort(key=_conflict_sort_key)
    return out


def _conflict_sort_key(c: Conflict) -> tuple[int, int, int, int, int]:
    """Return a stable lexicographic key for sorting conflicts deterministically."""
    kind = 0 if isinstance(c, VertexConflict) else 1
    return (c.agent_i, c.agent_j, c.t_i, c.t_j, kind)


def detect_conflicts_fast(plan: Plan, delta: int = 0) -> list[Conflict]:
    """Window-aware EXACT equivalent of :func:`detect_conflicts`.

    Returns the **same** set of conflicts as :func:`detect_conflicts` (and,
    after the final sort, the same list in the same order), but in roughly
    ``O(total_visits * local_window)`` instead of ``O(n^2 * makespan)``: it
    buckets occupancy by vertex (and traversals by undirected edge), sorts each
    bucket by time, and emits only the visit pairs that actually lie within
    ``delta`` via a sliding window — never enumerating slot-pairs guaranteed to
    be more than ``delta`` apart. This is **not** an approximation: ``delta`` is
    a hard set-membership boundary and every in-window pair is emitted.

    Occupancy is reproduced *exactly* as :meth:`Plan.vertex_visit`: each agent
    occupies ``vertex_visit(i, t)`` for every ``t in [0, makespan]`` — its
    explicit path cells for ``t < len(path)``, then its goal cell padded from
    ``len(path)`` through ``makespan`` (the goal-rest interval). Convention B
    (an agent leaving and re-entering a cell) is handled naturally: each visit
    is a separate ``(agent, t)`` bucket entry.

    Raises
    ------
    ValueError
        If ``delta`` is negative.
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    horizon = plan.makespan
    paths = {p.agent_id: p.vertices for p in plan.paths}
    agents = sorted(a.id for a in plan.agents)

    def occ(i: int, t: int) -> Cell:
        """vertex_visit(i, t): path cell, or goal-padded for t beyond the path."""
        verts = paths[i]
        return verts[t] if t < len(verts) else verts[-1]

    out: list[Conflict] = []

    # ---- vertex conflicts: bucket every (agent, t) occupancy by cell ----
    # vertex_visit scans t in [0, horizon]; goal-rest padding is included so
    # terminal occupancy participates exactly as in the original.
    vertex_visits: dict[Cell, list[tuple[int, int]]] = defaultdict(list)
    for i in agents:
        verts = paths[i]
        for t in range(len(verts)):
            vertex_visits[verts[t]].append((t, i))
        # Goal-rest: cell verts[-1] is also occupied at every t in (last, horizon].
        goal = verts[-1]
        for t in range(len(verts), horizon + 1):
            vertex_visits[goal].append((t, i))

    for cell, visits in vertex_visits.items():
        # Sort by (time, agent); a sliding window emits every cross-agent pair
        # with |t_a - t_b| <= delta exactly once, canonicalised to agent_i<agent_j.
        visits.sort()
        lo = 0
        for hi in range(len(visits)):
            t_b, a_b = visits[hi]
            while visits[lo][0] < t_b - delta:
                lo += 1
            for k in range(lo, hi):
                t_a, a_a = visits[k]
                if a_a == a_b:
                    continue
                # canonical: lower agent id is agent_i, carrying its own time.
                if a_a < a_b:
                    ci, ti, cj, tj = a_a, t_a, a_b, t_b
                else:
                    ci, ti, cj, tj = a_b, t_b, a_a, t_a
                out.append(
                    VertexConflict(agent_i=ci, agent_j=cj, vertex=cell, t_i=ti, t_j=tj)
                )

    # ---- edge/swap conflicts: bucket directed traversals by undirected edge ----
    # Agent i traverses u->v over [t, t+1] for t in [0, horizon) when occ(i,t) !=
    # occ(i,t+1). A swap is i: u->v and j: v->u with |t_i - t_j| <= delta.
    # Key each traversal by the frozenset {u, v}; within a bucket, a swap is a
    # pair whose directions are opposite.
    edge_moves: dict[frozenset[Cell], list[tuple[int, int, Cell, Cell]]] = defaultdict(list)
    for i in agents:
        for t in range(horizon):
            u = occ(i, t)
            v = occ(i, t + 1)
            if u == v:
                continue  # waiting; no traversal
            edge_moves[frozenset((u, v))].append((t, i, u, v))

    for moves in edge_moves.values():
        moves.sort()  # by (time, agent, u, v)
        m = len(moves)
        lo = 0
        for hi in range(m):
            t_b, a_b, u_b, v_b = moves[hi]
            while moves[lo][0] < t_b - delta:
                lo += 1
            for k in range(lo, hi):
                t_a, a_a, u_a, v_a = moves[k]
                if a_a == a_b:
                    continue
                # opposite directions: a's u->v must equal b's v->u
                if not (u_a == v_b and v_a == u_b):
                    continue
                # canonical: lower agent id is agent_i; (u, v) is its direction.
                if a_a < a_b:
                    ci, ti, ui, vi, cj, tj = a_a, t_a, u_a, v_a, a_b, t_b
                else:
                    ci, ti, ui, vi, cj, tj = a_b, t_b, u_b, v_b, a_a, t_a
                out.append(
                    EdgeConflict(agent_i=ci, agent_j=cj, u=ui, v=vi, t_i=ti, t_j=tj)
                )

    out.sort(key=_conflict_sort_key)
    return out
