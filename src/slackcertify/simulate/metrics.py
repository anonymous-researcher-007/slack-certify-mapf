"""Common reporting metrics for slack-certified plans."""

from __future__ import annotations

from slackcertify.core.plan import Plan
from slackcertify.utils.stats import wilson_ci as _wilson_ci

__all__ = ["makespan_overhead", "soc_overhead", "wilson_ci"]


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Re-export of :func:`slackcertify.utils.stats.wilson_ci`.

    Examples
    --------
    >>> lo, hi = wilson_ci(50, 100)
    >>> 0.0 <= lo <= 0.5 <= hi <= 1.0
    True
    """
    return _wilson_ci(k, n, alpha=alpha)


def soc_overhead(pi_prime: Plan, pi: Plan) -> float:
    """Return the relative sum-of-costs overhead of ``pi_prime`` over ``pi``.

    ``(pi_prime.SoC - pi.SoC) / pi.SoC``. When ``pi.SoC`` is zero the
    function returns the absolute difference instead so the result
    remains finite.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
    >>> p_short = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
    >>> p_long = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 0), (2, 0)])]
    >>> abs(soc_overhead(Plan.from_paths(a, p_long), Plan.from_paths(a, p_short)) - 0.5) < 1e-12
    True
    """
    if pi.sum_of_costs == 0:
        return float(pi_prime.sum_of_costs - pi.sum_of_costs)
    return (pi_prime.sum_of_costs - pi.sum_of_costs) / pi.sum_of_costs


def makespan_overhead(pi_prime: Plan, pi: Plan) -> float:
    """Return the relative makespan overhead of ``pi_prime`` over ``pi``.

    Same convention as :func:`soc_overhead` for the zero-baseline case.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
    >>> short = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
    >>> long_ = [Path(agent_id=0, vertices=[(0, 0), (0, 0), (1, 0)])]
    >>> makespan_overhead(Plan.from_paths(a, long_), Plan.from_paths(a, short))
    1.0
    """
    if pi.makespan == 0:
        return float(pi_prime.makespan - pi.makespan)
    return (pi_prime.makespan - pi.makespan) / pi.makespan
