"""Edge-conflict resolver.

For an :class:`EdgeConflict` ``(i, j, u, v, t_i, t_j)`` where agent ``i``
traverses ``u → v`` and agent ``j`` traverses ``v → u``, we shift the
*downstream* agent (the one with the larger traversal start time, ties
broken by ``agent_id``) so that the swap no longer overlaps.

The shift is implemented via :func:`slackcertify.repair.propagation.propagate_shift`
with the anchor set to the *destination* of the downstream agent's
traversal, i.e. the cell where the agent first arrives at the swap edge's
far end. This is equivalent to inserting waits at the agent's pre-swap
cell, which is what Algorithm 1 of the paper prescribes.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> from slackcertify.core.conflict import detect_conflicts, EdgeConflict
>>> agents = [Agent(id=0, start=(0, 0), goal=(1, 0)),
...           Agent(id=1, start=(1, 0), goal=(0, 0))]
>>> paths = [Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
...          Path(agent_id=1, vertices=[(1, 0), (0, 0)])]
>>> plan = Plan.from_paths(agents, paths)
>>> ec = next(c for c in detect_conflicts(plan) if isinstance(c, EdgeConflict))
>>> resolved = resolve_edge_conflict(plan, ec, delta=0, mode="bounded")
>>> any(isinstance(c, EdgeConflict) for c in detect_conflicts(resolved))
False
"""

from __future__ import annotations

from typing import Literal

from slackcertify.certify.probabilistic import _binom_diff_at_offset
from slackcertify.core.conflict import EdgeConflict
from slackcertify.core.plan import Plan
from slackcertify.repair.propagation import propagate_shift

__all__ = ["resolve_edge_conflict"]

_PROBABILISTIC_MAX_W: int = 4096


def resolve_edge_conflict(
    plan: Plan,
    c: EdgeConflict,
    delta: int | None,
    mode: Literal["bounded", "probabilistic"],
    p_d: float | None = None,
    per_conflict_budget: float | None = None,
    delta_window: int = 0,
) -> Plan:
    """Return a copy of ``plan`` with edge conflict ``c`` resolved.

    Parameters
    ----------
    plan
        Source plan; not mutated.
    c
        The edge conflict to resolve.
    delta
        Bounded-mode tolerance. Ignored in ``mode='probabilistic'``.
    mode
        Either ``'bounded'`` or ``'probabilistic'``.
    p_d, per_conflict_budget
        Probabilistic-mode parameters. Required when
        ``mode == 'probabilistic'``.

    Returns
    -------
    Plan
        A new plan in which the downstream agent of ``c`` waits ``w``
        steps at its pre-swap cell so the swap no longer overlaps under
        the chosen mode. When ``w == 0`` the input plan is returned
        unchanged (no copy).

    Raises
    ------
    ValueError
        If the requested mode's parameters are missing or invalid.
    """
    downstream_agent, anchor, t_down, t_other = _classify(c)

    if mode == "bounded":
        if delta is None or delta < 0:
            raise ValueError(f"bounded mode requires non-negative delta, got {delta}")
        gap = abs(t_down - t_other)
        if gap > delta:
            return plan
        w = delta - gap + 1
    elif mode == "probabilistic":
        if p_d is None or per_conflict_budget is None:
            raise ValueError("probabilistic mode requires both p_d and per_conflict_budget")
        if not (0.0 <= p_d <= 1.0):
            raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
        if per_conflict_budget < 0.0:
            raise ValueError(f"per_conflict_budget must be non-negative, got {per_conflict_budget}")
        w = _minimum_w_probabilistic(
            t_down, t_other, p_d, per_conflict_budget, delta_window=delta_window
        )
    else:
        raise ValueError(f"unsupported mode: {mode!r}")

    if w == 0:
        return plan
    return propagate_shift(plan, downstream_agent, anchor, w)


def _classify(c: EdgeConflict) -> tuple[int, tuple[int, int], int, int]:
    """Pick the downstream agent and the anchor cell for shifting it.

    Returns ``(downstream_agent_id, anchor_cell, t_downstream, t_other)``.
    """
    # Agent i goes u -> v; agent j goes v -> u. The *downstream* agent is
    # the one with the larger traversal-start time; ties go to the larger
    # agent id so the result is deterministic.
    if c.t_i > c.t_j or (c.t_i == c.t_j and c.agent_i > c.agent_j):
        # i is downstream; its destination is v.
        return c.agent_i, c.v, c.t_i, c.t_j
    # j is downstream; its destination is u.
    return c.agent_j, c.u, c.t_j, c.t_i


def _minimum_w_probabilistic(
    t_down: int,
    t_other: int,
    p_d: float,
    budget: float,
    delta_window: int = 0,
) -> int:
    """Minimum ``w`` so that shifted edge conflict's risk drops to ``budget``."""
    for w in range(_PROBABILISTIC_MAX_W + 1):
        new_t_down = t_down + w
        gap = abs(new_t_down - t_other)
        risk = _binom_diff_at_offset(
            t_smaller=min(new_t_down, t_other),
            t_larger=max(new_t_down, t_other),
            p_d=p_d,
            signed_gap=gap,
            delta_window=delta_window,
        )
        if risk <= budget:
            return w
    raise ValueError(
        f"unable to bring edge-conflict risk under {budget} within "
        f"{_PROBABILISTIC_MAX_W} wait steps"
    )
