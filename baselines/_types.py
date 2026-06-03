"""Shared lightweight types for the comparison-baseline wrappers.

Lives outside :mod:`slackcertify` because the baselines themselves are not
part of the public API; they are external comparison tools that the
experiment runners shell out to.

The :class:`BaselinePlan` dataclass is the canonical return type for every
baseline wrapper: it carries the raw :class:`~slackcertify.core.plan.Plan`
plus a free-form ``solver_used`` string used by the analysis layer to
distinguish baselines (``"kr_eecbs_k=2_w=1.2"``,
``"<nominal>+btpg_max"``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from slackcertify.core.plan import Plan

__all__ = ["BaselinePlan"]


@dataclass(frozen=True, slots=True)
class BaselinePlan:
    """A :class:`Plan` paired with the baseline-tag string that produced it."""

    plan: Plan
    solver_used: str
