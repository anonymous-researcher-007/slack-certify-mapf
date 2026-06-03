"""Conflict ordering strategies for the slack-certifier outer loop.

The repair loop processes one conflict at a time; the order in which
conflicts are picked materially changes the total amount of slack inserted.
Three strategies are exposed:

* :func:`topological_order` — the canonical paper ordering, sorting on
  ``(min(t_i, t_j), tpg_topological_rank)``.
* :func:`risk_descending_order` — most-dangerous-first, used in
  probabilistic mode.
* :func:`random_order` — shuffle, used in ablations.
"""

from __future__ import annotations

import numpy as np

from slackcertify.certify.probabilistic import per_conflict_risk
from slackcertify.core.conflict import Conflict, EdgeConflict, VertexConflict
from slackcertify.core.tpg import TPG, TPGNode

__all__ = ["random_order", "risk_descending_order", "topological_order"]


def topological_order(conflicts: list[Conflict], tpg: TPG) -> list[Conflict]:
    """Return ``conflicts`` sorted by ``(min(t_i, t_j), tpg_topological_rank)``.

    The TPG node used for the rank lookup is the one belonging to the
    *earlier* visiting agent of the conflict. Conflicts whose anchor
    node is missing from the TPG (which should not happen for a
    well-formed plan) sort to the end.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> from slackcertify.core.tpg import build_tpg
    >>> from slackcertify.core.conflict import detect_conflicts
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 1)),
    ...      Agent(id=1, start=(1, 3), goal=(1, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 1), (2, 1)]),
    ...      Path(agent_id=1, vertices=[(1, 3), (1, 2), (1, 1), (1, 0)])]
    >>> plan = Plan.from_paths(a, p)
    >>> ordered = topological_order(detect_conflicts(plan), build_tpg(plan))
    >>> all(min(c.t_i, c.t_j) <= min(d.t_i, d.t_j)
    ...     for c, d in zip(ordered, ordered[1:]))
    True
    """
    rank_of = {n: i for i, n in enumerate(tpg.topological_order())}
    sentinel = len(rank_of)

    def _key(c: Conflict) -> tuple[int, int, int, int]:
        """Sort key: ``(min t, TPG rank, agent_i, agent_j)`` for stable topo order."""
        anchor = _earlier_anchor_node(c)
        rank = rank_of.get(anchor, sentinel)
        return (min(c.t_i, c.t_j), rank, c.agent_i, c.agent_j)

    return sorted(conflicts, key=_key)


def risk_descending_order(conflicts: list[Conflict], p_d: float) -> list[Conflict]:
    """Return ``conflicts`` sorted by descending per-conflict risk.

    Ties are broken by ``(min(t_i, t_j), agent_i, agent_j)`` ascending.

    Examples
    --------
    >>> from slackcertify.core.conflict import VertexConflict
    >>> c1 = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3)
    >>> c2 = VertexConflict(agent_i=0, agent_j=2, vertex=(0, 0), t_i=2, t_j=2)
    >>> # c1 has positive risk; c2 has zero (gap=0).
    >>> [c.agent_j for c in risk_descending_order([c2, c1], p_d=0.3)]
    [1, 2]
    """
    if not (0.0 <= p_d <= 1.0):
        raise ValueError(f"p_d must lie in [0, 1], got {p_d}")

    def _key(c: Conflict) -> tuple[float, int, int, int]:
        """Sort key: ``(-risk, min t, agent_i, agent_j)`` for risk-descending order."""
        return (-per_conflict_risk(c, p_d), min(c.t_i, c.t_j), c.agent_i, c.agent_j)

    return sorted(conflicts, key=_key)


def random_order(conflicts: list[Conflict], rng: np.random.Generator) -> list[Conflict]:
    """Return ``conflicts`` in a random order, drawn from ``rng``.

    Examples
    --------
    >>> import numpy as np
    >>> from slackcertify.core.conflict import VertexConflict
    >>> cs = [VertexConflict(0, 1, (i, 0), i, i + 1) for i in range(4)]
    >>> permuted = random_order(cs, np.random.default_rng(0))
    >>> sorted(c.t_i for c in permuted) == [0, 1, 2, 3]
    True
    """
    out = list(conflicts)
    rng.shuffle(out)
    return out


def _earlier_anchor_node(c: Conflict) -> TPGNode:
    """Return the TPG node of the earlier-visiting agent of ``c``."""
    if c.t_i <= c.t_j:
        early_agent, early_t = c.agent_i, c.t_i
        early_cell = c.vertex if isinstance(c, VertexConflict) else c.u
    else:
        early_agent, early_t = c.agent_j, c.t_j
        early_cell = c.vertex if isinstance(c, VertexConflict) else c.v
    return TPGNode(early_agent, early_cell, early_t)


# Re-export EdgeConflict for use in dependent modules.
_ = EdgeConflict
