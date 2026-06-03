"""Delay models used by the slack certifier and the simulator.

Three families are provided out of the box:

* :class:`BoundedDelayModel` — worst-case cumulative-Δ bounded delay.
* :class:`BernoulliDelayModel` — i.i.d. per-step Bernoulli delay with a
  closed-form Binomial cumulative distribution.
* :class:`PersistentDelayModel` — MAPF-DP-style rare-trigger /
  long-stall persistent delay.

Each model exposes ``sample(plan, rng) -> dict[agent_id, list[int]]``
returning a per-step delay vector per agent, and (for the bounded model)
a deterministic ``worst_case_against`` adversary. The worst-case
schedule for any model can also be obtained via
:meth:`WorstCaseAdversary.for_model`.

A small calibration helper —
:func:`calibrate_bernoulli_to_persistent` — converts a persistent
specification into the equivalent Bernoulli ``p_d``.
"""

from __future__ import annotations

from slackcertify.delay.adversarial import HeuristicAdversaryDelay
from slackcertify.delay.adversary import Adversary, WorstCaseAdversary
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.delay.calibration import calibrate_bernoulli_to_persistent
from slackcertify.delay.persistent import PersistentDelayModel

__all__ = [
    "Adversary",
    "BernoulliDelayModel",
    "BoundedDelayModel",
    "HeuristicAdversaryDelay",
    "PersistentDelayModel",
    "WorstCaseAdversary",
    "calibrate_bernoulli_to_persistent",
]
