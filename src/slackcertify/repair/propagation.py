"""Wait-insertion + downstream-shift primitive.

:func:`propagate_shift` is the workhorse used by both the vertex-conflict
resolver and :mod:`slackcertify.repair.edge_handler`. It returns a *new*
:class:`Plan` (no input mutation) in which the named agent waits for ``w``
extra steps just before its first arrival at ``before_vertex``; every
downstream cell in that agent's path is therefore shifted by ``w``
time-steps. Other agents' paths are untouched.

If the agent first reaches ``before_vertex`` at index ``0`` (i.e. the
starting cell *is* ``before_vertex``), the waits are prepended at the
start.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> shifted = propagate_shift(Plan.from_paths(a, p), 0, (2, 0), 2)
>>> shifted.paths[0].vertices
[(0, 0), (1, 0), (1, 0), (1, 0), (2, 0)]
"""

from __future__ import annotations

from slackcertify.core.graph import Cell
from slackcertify.core.plan import Path, Plan

__all__ = ["propagate_shift"]


def propagate_shift(plan: Plan, agent_id: int, before_vertex: Cell, w: int) -> Plan:
    """Return a copy of ``plan`` with ``w`` waits inserted before ``before_vertex``.

    Parameters
    ----------
    plan
        Source plan; not mutated.
    agent_id
        The agent whose path is shifted.
    before_vertex
        The cell whose first occurrence in ``agent_id``'s path marks the
        insertion point.
    w
        Number of wait steps to insert. ``w == 0`` returns the input
        plan unchanged; negative values raise :class:`ValueError`.

    Returns
    -------
    Plan
        A new plan with ``agent_id``'s path lengthened by ``w`` steps;
        every other path is identical to the input.

    Raises
    ------
    ValueError
        If ``w < 0``, ``agent_id`` is not in ``plan``, or ``agent_id``'s
        path never visits ``before_vertex``.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(1, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(1, 0), (0, 0)])]
    >>> # before_vertex is the start cell -> waits are prepended.
    >>> propagate_shift(Plan.from_paths(a, p), 0, (1, 0), 1).paths[0].vertices
    [(1, 0), (1, 0), (0, 0)]
    """
    if w < 0:
        raise ValueError(f"w must be non-negative, got {w}")
    if w == 0:
        return plan

    if agent_id not in {a.id for a in plan.agents}:
        raise ValueError(f"unknown agent_id={agent_id}")

    new_paths: list[Path] = []
    inserted = False
    for path in plan.paths:
        if path.agent_id != agent_id:
            new_paths.append(path)
            continue
        verts = list(path.vertices)
        try:
            idx = verts.index(before_vertex)
        except ValueError as exc:
            raise ValueError(
                f"agent {agent_id} never visits {before_vertex!r} in its path"
            ) from exc
        if idx == 0:
            wait_cell = verts[0]
            new_verts = [wait_cell] * w + verts
        else:
            wait_cell = verts[idx - 1]
            new_verts = verts[:idx] + [wait_cell] * w + verts[idx:]
        new_paths.append(Path(agent_id=agent_id, vertices=new_verts))
        inserted = True

    if not inserted:
        # Should be impossible given the agent_id check above; guard anyway.
        raise ValueError(f"no path with agent_id={agent_id} found in plan")

    return Plan.from_paths(plan.agents, new_paths)
