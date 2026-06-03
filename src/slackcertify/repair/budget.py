"""Per-conflict ε-budget allocation strategies.

In probabilistic mode the certifier guarantees the union-bound risk is at most
``ε``. The two helpers here split that ``ε`` across the conflicts so each one
gets its own per-conflict budget:

* :func:`uniform_budget` — every conflict gets the same share.
* :func:`risk_proportional_budget` — share is proportional to the
  conflict's *raw* (pre-repair) risk, so high-risk conflicts get more
  slack to absorb.

Both strategies return a budget that the per-conflict risk must stay
below; their sum across conflicts is at most ``ε``.

Examples
--------
>>> uniform_budget(0.10, 5)
0.02
"""

from __future__ import annotations

from slackcertify.certify.probabilistic import per_conflict_risk
from slackcertify.core.conflict import Conflict

__all__ = ["risk_proportional_budget", "uniform_budget"]


def uniform_budget(epsilon: float, n_conflicts: int) -> float:
    """Return the per-conflict share ``epsilon / n_conflicts``.

    With ``n_conflicts == 0`` the function returns ``+inf`` so any
    risk passes the trivial test.

    Examples
    --------
    >>> uniform_budget(0.05, 0) == float("inf")
    True
    >>> uniform_budget(0.1, 4)
    0.025
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if n_conflicts < 0:
        raise ValueError(f"n_conflicts must be non-negative, got {n_conflicts}")
    if n_conflicts == 0:
        return float("inf")
    return epsilon / n_conflicts


def risk_proportional_budget(
    epsilon: float, conflicts: list[Conflict], p_d: float
) -> dict[Conflict, float]:
    """Return per-conflict budgets proportional to each conflict's raw risk.

    A conflict ``c`` whose raw :func:`per_conflict_risk` is ``r_c``
    receives ``epsilon * r_c / sum(r)``. If every raw risk is zero the
    function falls back to :func:`uniform_budget`.

    The sum of the returned values is guaranteed to be at most
    ``epsilon`` (exactly ``epsilon`` whenever any raw risk is positive).

    Examples
    --------
    >>> from slackcertify.core.conflict import VertexConflict
    >>> c1 = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3)
    >>> c2 = VertexConflict(agent_i=0, agent_j=2, vertex=(1, 1), t_i=4, t_j=5)
    >>> b = risk_proportional_budget(0.1, [c1, c2], p_d=0.0)
    >>> abs(sum(b.values()) - 0.1) < 1e-12
    True
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if not (0.0 <= p_d <= 1.0):
        raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
    if not conflicts:
        return {}

    raw = {c: per_conflict_risk(c, p_d) for c in conflicts}
    total = sum(raw.values())
    if total <= 0.0:
        share = uniform_budget(epsilon, len(conflicts))
        return dict.fromkeys(conflicts, share)
    return {c: epsilon * (r / total) for c, r in raw.items()}
