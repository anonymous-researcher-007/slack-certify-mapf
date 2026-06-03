"""Worst-case adversary factory.

Each delay model has a corresponding "worst-case adversary" — a deterministic
schedule that is, in some natural sense, the hardest the model can throw at a
given plan. :class:`WorstCaseAdversary` packages those constructors behind a
single :meth:`for_model` factory so the certifier can request an adversary
without caring which model it came from.

Conventions:

* :class:`~slackcertify.delay.bounded.BoundedDelayModel` →
  the greedy schedule from
  :meth:`~slackcertify.delay.bounded.BoundedDelayModel.worst_case_against`.
* :class:`~slackcertify.delay.bernoulli.BernoulliDelayModel` →
  every step delays (the ``p_d → 1`` limit).
* :class:`~slackcertify.delay.persistent.PersistentDelayModel` →
  one back-to-back stall episode of length ``max_len`` starting at ``t=0``,
  repeated until the path ends.

Examples
--------
>>> from slackcertify.delay.bounded import BoundedDelayModel
>>> adv = WorstCaseAdversary.for_model(BoundedDelayModel(delta=2))
>>> isinstance(adv, WorstCaseAdversary)
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from slackcertify.core.conflict import Conflict
from slackcertify.core.plan import Plan
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.delay.persistent import PersistentDelayModel

__all__ = ["Adversary", "WorstCaseAdversary"]


Adversary = Callable[[Plan, list[Conflict]], dict[int, list[int]]]


@dataclass(frozen=True, slots=True)
class WorstCaseAdversary:
    """Callable wrapping a model-specific worst-case schedule constructor."""

    fn: Adversary

    def __call__(self, plan: Plan, conflicts: list[Conflict]) -> dict[int, list[int]]:
        """Invoke the wrapped adversary callable on ``(plan, conflicts)``."""
        return self.fn(plan, conflicts)

    @classmethod
    def for_model(
        cls,
        model: BoundedDelayModel | BernoulliDelayModel | PersistentDelayModel,
    ) -> WorstCaseAdversary:
        """Return the worst-case adversary corresponding to ``model``.

        Examples
        --------
        >>> WorstCaseAdversary.for_model(BernoulliDelayModel(p_d=0.1))(
        ...     __import__("slackcertify").Plan.from_paths([], []), []
        ... )
        {}
        """
        if isinstance(model, BoundedDelayModel):
            return cls(fn=model.worst_case_against)
        if isinstance(model, BernoulliDelayModel):
            return cls(fn=_bernoulli_worst_case)
        if isinstance(model, PersistentDelayModel):
            return cls(fn=_persistent_worst_case_factory(model))
        raise TypeError(f"unsupported delay model: {type(model).__name__}")


def _bernoulli_worst_case(
    plan: Plan,
    conflicts: list[Conflict],  # noqa: ARG001 - signature
) -> dict[int, list[int]]:
    """Every step delays — the ``p_d → 1`` limit of the Bernoulli model."""
    return {p.agent_id: [1] * p.makespan for p in plan.paths}


def _persistent_worst_case_factory(model: PersistentDelayModel) -> Adversary:
    """Back-to-back ``max_len`` stalls covering the entire schedule."""

    def fn(
        plan: Plan,
        conflicts: list[Conflict],  # noqa: ARG001 - signature
    ) -> dict[int, list[int]]:
        """Adversary closure: every agent stalls every step (worst case)."""
        out: dict[int, list[int]] = {}
        for p in plan.paths:
            out[p.agent_id] = [1] * p.makespan
        # The single longest possible stall (max_len) is just all-ones up to
        # the path makespan; larger schedules are bounded by the path itself,
        # so further repeats add no realised delay.
        _ = model.max_len  # keep the parameter materially referenced
        return out

    return fn
