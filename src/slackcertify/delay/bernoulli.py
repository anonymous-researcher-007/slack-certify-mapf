"""Per-step independent Bernoulli delay model.

Each agent at each step independently delays one tick with probability
``p_d``. Cumulative delay after ``t`` steps is therefore
``Binomial(t, p_d)``; the closed form is what the probabilistic
certifier of §IV exploits.

Examples
--------
>>> import numpy as np
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> rng = np.random.default_rng(0)
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> sched = BernoulliDelayModel(p_d=0.5).sample(Plan.from_paths(a, p), rng)
>>> len(sched[0])
2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binom

from slackcertify.core.plan import Plan

__all__ = ["BernoulliDelayModel"]


@dataclass(frozen=True, slots=True)
class BernoulliDelayModel:
    """I.i.d. Bernoulli per-step delay model.

    Parameters
    ----------
    p_d
        Per-step delay probability. Must lie in ``[0, 1)``.
    """

    p_d: float

    def __post_init__(self) -> None:
        """Validate that ``p_d`` lies in ``[0, 1)``."""
        if not (0.0 <= self.p_d < 1.0):
            raise ValueError(f"p_d must lie in [0, 1), got {self.p_d}")

    def sample(self, plan: Plan, rng: np.random.Generator) -> dict[int, list[int]]:
        """Draw an i.i.d. Bernoulli(p_d) delay per step per agent.

        Examples
        --------
        >>> import numpy as np
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> rng = np.random.default_rng(7)
        >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
        >>> BernoulliDelayModel(p_d=0.3).sample(Plan.from_paths(a, p), rng)
        {0: []}
        """
        out: dict[int, list[int]] = {}
        for path in plan.paths:
            steps = path.makespan
            if steps == 0:
                out[path.agent_id] = []
                continue
            draws = rng.random(size=steps) < self.p_d
            out[path.agent_id] = [int(x) for x in draws]
        return out

    def cumulative_delay_distribution(self, t: int) -> Any:  # noqa: ANN401 - scipy frozen
        """Return the ``Binomial(t, p_d)`` frozen distribution.

        Examples
        --------
        >>> dist = BernoulliDelayModel(p_d=0.3).cumulative_delay_distribution(10)
        >>> round(float(dist.mean()), 6)
        3.0
        """
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        return binom(n=t, p=self.p_d)
