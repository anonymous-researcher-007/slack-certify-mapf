"""Bounded-delay certifier (Definition 1).

A plan is **Δ-certified** iff every pair of agents that could in principle
collide does so only outside a Δ-window — i.e. for every shared vertex
``v`` and every reverse-traversed edge, the visit times satisfy
``|t_i - t_j| > Δ``. Equivalently, :func:`detect_conflicts` returns the
empty list at tolerance ``Δ``.

The two helpers exposed here are pure and do not mutate their inputs.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> is_delta_certified(Plan.from_paths(a, p), delta=0)
True
"""

from __future__ import annotations

from slackcertify.core.conflict import Conflict, detect_conflicts
from slackcertify.core.plan import Plan

__all__ = ["is_delta_certified", "unsafe_pairs_bounded"]


def is_delta_certified(plan: Plan, delta: int) -> bool:
    """Return ``True`` iff ``plan`` has no Δ-disjointness violations.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0)),
    ...      Agent(id=1, start=(1, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
    ...      Path(agent_id=1, vertices=[(1, 0), (0, 0)])]
    >>> is_delta_certified(Plan.from_paths(a, p), delta=0)
    False
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    return not detect_conflicts(plan, delta=delta)


def unsafe_pairs_bounded(plan: Plan, delta: int) -> list[Conflict]:
    """Return every (V) and (E) Δ-disjointness violation in ``plan``.

    This is a thin alias for :func:`detect_conflicts` that documents the
    bounded-certifier semantics: a returned conflict is exactly an
    instance of "Definition 1 fails for this pair".

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0)),
    ...      Agent(id=1, start=(1, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
    ...      Path(agent_id=1, vertices=[(1, 0), (0, 0)])]
    >>> len(unsafe_pairs_bounded(Plan.from_paths(a, p), delta=0)) >= 1
    True
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    return detect_conflicts(plan, delta=delta)
