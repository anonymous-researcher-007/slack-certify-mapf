"""Cheap structural pre-check for wait-insertion feasibility.

A head-on swap on a topology where both agents traverse the same
cell set in opposite directions cannot be resolved by waits alone —
no matter how many wait actions we insert, neither agent has a
side-bay cell to step aside into. The :func:`is_wait_resolvable`
predicate detects exactly this case in ``O(|conflicts| + Σ_i |path_i|)``
time so both the greedy certifier and the ILP solver can fail fast
with a clear "structurally infeasible" diagnostic instead of
spinning their wheels.

Vertex conflicts are always wait-resolvable in principle (one agent
can simply wait for the other to clear the cell), so the predicate
inspects only :class:`~slackcertify.core.conflict.EdgeConflict`
instances.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> from slackcertify.core.conflict import detect_conflicts
>>> a = [Agent(id=0, start=(0, 0), goal=(0, 3)),
...      Agent(id=1, start=(0, 3), goal=(0, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (0, 1), (0, 2), (0, 3)]),
...      Path(agent_id=1, vertices=[(0, 3), (0, 2), (0, 1), (0, 0)])]
>>> plan = Plan.from_paths(a, p)
>>> ok, bad = is_wait_resolvable(plan, detect_conflicts(plan, delta=1))
>>> ok, len(bad)
(False, 1)
"""

from __future__ import annotations

from slackcertify.core.conflict import Conflict, EdgeConflict
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Plan

__all__ = ["is_wait_resolvable"]


def is_wait_resolvable(
    plan: Plan,
    conflicts: list[Conflict],
    graph: GridGraph | None = None,
) -> tuple[bool, list[Conflict]]:
    """Return ``(True, [])`` if every conflict is wait-resolvable.

    Otherwise returns ``(False, [unresolvable_conflicts])``.

    Predicate: for every :class:`EdgeConflict` (head-on swap), at
    least one of the two agents must have a path cell the other
    agent never visits. If both agents traverse exactly the same
    cell set, neither has a side-bay to step aside into, so the
    swap is structurally unresolvable by wait insertion alone.

    The ``graph`` parameter is reserved for future graph-aware
    extensions (e.g. checking whether a globally-free side cell
    exists outside both agents' paths); the current implementation
    does not consult it.

    Parameters
    ----------
    plan
        Plan whose paths supply the per-agent cell sets.
    conflicts
        Conflict list to check; pass the output of
        :func:`detect_conflicts` for the input plan + tolerance.
    graph
        Reserved; ignored.

    Returns
    -------
    (bool, list[Conflict])
        ``(True, [])`` iff every conflict is wait-resolvable;
        otherwise ``(False, list of unresolvable EdgeConflicts)``.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> from slackcertify.core.conflict import detect_conflicts
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 1)),
    ...      Agent(id=1, start=(1, 3), goal=(1, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 1), (2, 1)]),
    ...      Path(agent_id=1, vertices=[(1, 3), (1, 2), (1, 1), (1, 0)])]
    >>> plan = Plan.from_paths(a, p)
    >>> is_wait_resolvable(plan, detect_conflicts(plan, delta=0))
    (True, [])
    """
    _ = graph  # reserved for future graph-aware extensions
    agent_cells: dict[int, frozenset[tuple[int, int]]] = {
        path.agent_id: frozenset(path.vertices) for path in plan.paths
    }

    unresolvable: list[Conflict] = []
    for c in conflicts:
        if not isinstance(c, EdgeConflict):
            continue
        cells_i = agent_cells.get(c.agent_i)
        cells_j = agent_cells.get(c.agent_j)
        if cells_i is None or cells_j is None:
            continue
        # Both subset relations holding is equivalent to set equality;
        # we keep the spec wording for the audit trail.
        if cells_i.issubset(cells_j) and cells_j.issubset(cells_i):
            unresolvable.append(c)
    return len(unresolvable) == 0, unresolvable
