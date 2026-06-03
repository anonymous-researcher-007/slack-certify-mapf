"""MAPF-DP–style persistent delay model.

At every timestep an agent independently rolls ``p_trigger``; on success
the agent stalls for ``L ~ Uniform[min_len, max_len]`` consecutive
timesteps. While a stall is in progress the agent does not roll for new
triggers (triggers cannot overlap).

Examples
--------
>>> import numpy as np
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> rng = np.random.default_rng(0)
>>> a = [Agent(id=0, start=(0, 0), goal=(5, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])]
>>> sched = PersistentDelayModel(p_trigger=0.5, min_len=2, max_len=4).sample(
...     Plan.from_paths(a, p), rng
... )
>>> all(d in (0, 1) for d in sched[0])
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slackcertify.core.plan import Plan

__all__ = ["PersistentDelayModel"]


@dataclass(frozen=True, slots=True)
class PersistentDelayModel:
    """Persistent delay model: rare-trigger, long stall.

    Parameters
    ----------
    p_trigger
        Per-step probability of starting a new stall episode.
    min_len, max_len
        Inclusive bounds on the stall length, drawn uniformly per
        episode.
    """

    p_trigger: float
    min_len: int = 10
    max_len: int = 20

    def __post_init__(self) -> None:
        """Validate ``p_trigger``, ``min_len``, and ``max_len`` invariants."""
        if not (0.0 <= self.p_trigger < 1.0):
            raise ValueError(f"p_trigger must lie in [0, 1), got {self.p_trigger}")
        if self.min_len < 1:
            raise ValueError(f"min_len must be >= 1, got {self.min_len}")
        if self.max_len < self.min_len:
            raise ValueError(f"max_len ({self.max_len}) must be >= min_len ({self.min_len})")

    def sample(self, plan: Plan, rng: np.random.Generator) -> dict[int, list[int]]:
        """Sample a per-step delay schedule from the persistent process.

        Examples
        --------
        >>> import numpy as np
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> rng = np.random.default_rng(2)
        >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
        >>> sched = PersistentDelayModel(
        ...     p_trigger=0.0, min_len=1, max_len=1
        ... ).sample(Plan.from_paths(a, p), rng)
        >>> sched
        {0: [0, 0]}
        """
        out: dict[int, list[int]] = {}
        for path in plan.paths:
            steps = path.makespan
            schedule = [0] * steps
            stall_remaining = 0
            for t in range(steps):
                if stall_remaining > 0:
                    schedule[t] = 1
                    stall_remaining -= 1
                    continue
                if float(rng.random()) < self.p_trigger:
                    length = int(rng.integers(low=self.min_len, high=self.max_len + 1))
                    schedule[t] = 1
                    stall_remaining = length - 1
            out[path.agent_id] = schedule
        return out
