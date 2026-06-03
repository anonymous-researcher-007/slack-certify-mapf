"""Hypothesis strategies and a delay-aware simulator for the property tests.

The strategies are intentionally narrow: the property tests rely on plans that
are nominally conflict-free *and* exercise the certifier (i.e. they share at
least one vertex at staggered nominal times). The "cross plan" topology
satisfies both constraints by construction:

* two agents on a square open grid,
* one walks east-to-west through the centre, the other south-to-north,
* they share only the centre cell ``(mid, mid)``,
* their nominal visit times to that cell are configurable via the offsets,
* paths are 4-connected shortest lines.

A small multi-tick simulator (:func:`simulate_with_delays`) reconstructs each
agent's wall-clock occupancy intervals from a per-step delay vector and
returns ``True`` iff no two agents ever share a vertex (i.e. no realised
collision).

Examples
--------
>>> plan = make_cross_plan(grid_size=10, offsets=(2, 3))
>>> {a.id for a in plan.agents}
{0, 1}
>>> simulate_with_delays(plan, {0: [0] * plan.paths[0].makespan,
...                             1: [0] * plan.paths[1].makespan})
True
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from hypothesis import strategies as st

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Path, Plan

__all__ = [
    "GRID_SIZE",
    "cross_plans",
    "grid_graphs",
    "make_cross_plan",
    "plans_on_graph",
    "simulate_with_delays",
]


GRID_SIZE: int = 10


def grid_graphs(min_size: int = 3, max_size: int = 8) -> st.SearchStrategy[GridGraph]:
    """Strategy that draws an ``n x n`` open (no-obstacle) grid graph.

    Examples
    --------
    >>> import hypothesis
    >>> g = grid_graphs(3, 5).example()  # doctest: +SKIP
    """
    return st.builds(
        lambda n: GridGraph(width=n, height=n, obstacles=set()),
        st.integers(min_value=min_size, max_value=max_size),
    )


def make_cross_plan(grid_size: int, offsets: tuple[int, int]) -> Plan:
    """Construct a 2-agent cross plan on a ``grid_size`` square open grid.

    Agent 0 walks east→west through the middle column ``mid``;
    agent 1 walks south→north through the middle row.
    The shared cell ``(mid, mid)`` is visited at nominal times
    ``offsets[0]`` and ``offsets[1]`` respectively. The two paths share
    no other vertex.
    """
    if not 1 <= offsets[0] <= grid_size // 2 or not 1 <= offsets[1] <= grid_size // 2:
        raise ValueError(f"offsets {offsets} must lie in [1, grid_size // 2]")
    mid = grid_size // 2

    # Agent 0 — east→west on row `mid`.
    a0_start = (mid + offsets[0], mid)
    a0_goal = (mid - offsets[0], mid)
    a0_verts = [(mid + offsets[0] - k, mid) for k in range(2 * offsets[0] + 1)]

    # Agent 1 — south→north on column `mid`.
    a1_start = (mid, mid + offsets[1])
    a1_goal = (mid, mid - offsets[1])
    a1_verts = [(mid, mid + offsets[1] - k) for k in range(2 * offsets[1] + 1)]

    agents = [
        Agent(id=0, start=a0_start, goal=a0_goal),
        Agent(id=1, start=a1_start, goal=a1_goal),
    ]
    paths = [
        Path(agent_id=0, vertices=a0_verts),
        Path(agent_id=1, vertices=a1_verts),
    ]
    return Plan.from_paths(agents, paths)


def cross_plans(grid_size: int = GRID_SIZE) -> st.SearchStrategy[Plan]:
    """Strategy that draws a cross plan with random offsets.

    The offsets are constrained so the goals stay inside the grid and
    the shared-cell visit times are at least 1.

    Examples
    --------
    >>> from hypothesis import given
    >>> @given(plan=cross_plans())
    ... def test_plan_has_two_agents(plan):
    ...     assert len(plan.agents) == 2
    """
    half = max(2, grid_size // 2)  # need at least two distinct offsets
    pair = st.tuples(
        st.integers(min_value=1, max_value=half),
        st.integers(min_value=1, max_value=half),
    ).filter(lambda ab: ab[0] != ab[1])
    return st.builds(lambda ab: make_cross_plan(grid_size, ab), pair)


def plans_on_graph(
    graph: GridGraph,
    n_agents: st.SearchStrategy[int] | None = None,
) -> st.SearchStrategy[Plan]:
    """Strategy returning plans on ``graph``.

    The current implementation defers to :func:`cross_plans`; the
    ``graph`` argument fixes the side length and ``n_agents`` is
    accepted for API parity with future per-graph strategies.
    """
    _ = n_agents  # ignored in the cross-plan implementation
    return cross_plans(grid_size=max(graph.width, graph.height))


# ---------------------------------------------------------------- simulator


def simulate_with_delays(plan: Plan, delays: dict[int, list[int]]) -> bool:
    """Return ``True`` iff ``plan`` is collision-free under ``delays``.

    Reconstructs each agent's wall-clock occupancy intervals from its
    per-step delay vector and walks the joint timeline tick-by-tick,
    asserting no two agents ever share a vertex (vertex collision) or
    reverse-traverse the same edge (head-on swap collision).

    Parameters
    ----------
    plan
        The plan being executed.
    delays
        Per-agent integer delay vectors (length equals path makespan).
        Missing agents are treated as having all-zero delays.

    Returns
    -------
    bool
        ``True`` if no collision is realised; ``False`` otherwise.

    Examples
    --------
    >>> p = make_cross_plan(10, (2, 3))
    >>> simulate_with_delays(p, {0: [0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0]})
    True
    """
    agent_intervals: dict[int, list[tuple[int, float, tuple[int, int]]]] = {}
    horizon_finite = 0
    for path in plan.paths:
        ds = list(delays.get(path.agent_id, [0] * path.makespan))
        intervals: list[tuple[int, float, tuple[int, int]]] = []
        cumulative = 0
        for k, v in enumerate(path.vertices):
            t_k = k + cumulative
            if k < len(path.vertices) - 1:
                d_k = ds[k] if k < len(ds) else 0
                cumulative += d_k
                t_kp1 = (k + 1) + cumulative
                intervals.append((t_k, t_kp1 - 1, v))
                horizon_finite = max(horizon_finite, t_kp1 - 1)
            else:
                intervals.append((t_k, math.inf, v))
                horizon_finite = max(horizon_finite, t_k)
        agent_intervals[path.agent_id] = intervals

    horizon = horizon_finite + 2
    agent_ids = sorted(agent_intervals)

    def pos_at(agent_id: int, t: int) -> tuple[int, int] | None:
        for start, end, v in agent_intervals[agent_id]:
            if start <= t <= end:
                return v
        return None

    for t in range(0, horizon + 1):
        cells: dict[tuple[int, int], int] = {}
        for a in agent_ids:
            v = pos_at(a, t)
            if v is None:
                continue
            if v in cells:
                return False
            cells[v] = a

        # Edge collision detection: two agents that swap cells between t and t+1.
        for i_idx, a in enumerate(agent_ids):
            for b in agent_ids[i_idx + 1 :]:
                va_t = pos_at(a, t)
                vb_t = pos_at(b, t)
                va_t1 = pos_at(a, t + 1)
                vb_t1 = pos_at(b, t + 1)
                if None in (va_t, vb_t, va_t1, vb_t1):
                    continue
                if va_t != va_t1 and vb_t != vb_t1 and va_t == vb_t1 and vb_t == va_t1:
                    return False
    return True


def all_zero_delays(plan: Plan) -> dict[int, list[int]]:
    """Return a no-delay schedule sized to match every agent's path.

    Examples
    --------
    >>> p = make_cross_plan(10, (1, 1))
    >>> {k: len(v) for k, v in all_zero_delays(p).items()}
    {0: 2, 1: 2}
    """
    return {p.agent_id: [0] * p.makespan for p in plan.paths}


def initial_conflict_count(plan: Plan, delta: int) -> int:
    """Return the number of unsafe pairs in ``plan`` at the given ``delta``.

    A small helper used by :mod:`tests.property.test_termination` to
    compute the theoretical outer-round bound.
    """
    from slackcertify.core.conflict import detect_conflicts

    return len(detect_conflicts(plan, delta=delta))


# Make `Iterable` importable from this module for callers that want to
# spell the same type alias.
_ = Iterable
