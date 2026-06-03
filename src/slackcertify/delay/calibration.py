"""Calibration helpers between delay models.

The persistent (MAPF-DP) delay model is parameterised by trigger probability
and stall-length bounds, while the Bernoulli model is parameterised by a
single per-step delay probability. To answer §V RQ5 ("how does Slack-Certify
fare when calibrated against an equivalent Bernoulli baseline?") we need a
function that maps a persistent specification to its empirical per-step
delay rate.

Examples
--------
>>> import numpy as np
>>> from slackcertify.delay.persistent import PersistentDelayModel
>>> rng = np.random.default_rng(0)
>>> p_d = calibrate_bernoulli_to_persistent(
...     PersistentDelayModel(p_trigger=0.05, min_len=2, max_len=4),
...     plan_length=50,
...     rng=rng,
... )
>>> 0.0 <= p_d <= 1.0
True
"""

from __future__ import annotations

import numpy as np

from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.persistent import PersistentDelayModel

__all__ = ["calibrate_bernoulli_to_persistent"]

_NUM_ROLLOUTS: int = 200


def calibrate_bernoulli_to_persistent(
    persistent: PersistentDelayModel,
    plan_length: int,
    rng: np.random.Generator,
) -> float:
    """Return the empirical per-step blocking rate of ``persistent``.

    Runs ``M = 200`` independent rollouts of a single-agent plan of
    length ``plan_length`` against ``persistent`` and returns
    ``total_delay_steps / (M * plan_length)``. The result is suitable
    as a ``p_d`` surrogate for the equivalent Bernoulli model.

    Parameters
    ----------
    persistent
        The persistent delay model to calibrate against.
    plan_length
        Number of timesteps in the synthetic single-agent plan; must
        be positive.
    rng
        NumPy random generator used to draw rollouts.

    Returns
    -------
    float
        Empirical per-step blocking rate in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from slackcertify.delay.persistent import PersistentDelayModel
    >>> rng = np.random.default_rng(42)
    >>> p_d = calibrate_bernoulli_to_persistent(
    ...     PersistentDelayModel(p_trigger=0.0, min_len=1, max_len=1),
    ...     plan_length=20,
    ...     rng=rng,
    ... )
    >>> p_d
    0.0
    """
    if plan_length <= 0:
        raise ValueError(f"plan_length must be positive, got {plan_length}")

    # Synthetic single-agent straight-line plan of the requested length.
    vertices = [(i, 0) for i in range(plan_length + 1)]
    plan = Plan.from_paths(
        agents=[Agent(id=0, start=vertices[0], goal=vertices[-1])],
        paths=[Path(agent_id=0, vertices=vertices)],
    )

    total_delay = 0
    for _ in range(_NUM_ROLLOUTS):
        sched = persistent.sample(plan, rng)
        total_delay += sum(sched[0])
    return total_delay / (_NUM_ROLLOUTS * plan_length)
