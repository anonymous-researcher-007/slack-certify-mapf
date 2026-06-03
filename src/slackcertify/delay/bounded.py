"""Bounded per-agent delay model.

The bounded model captures the worst-case "every agent suffers at most ``Δ``
total stall steps" assumption used by the slack certifier in §III. A *delay
schedule* is a per-agent vector of non-negative integer per-step delays; a
schedule is *feasible* iff its cumulative sum never exceeds ``Δ``.

Sampling. :meth:`BoundedDelayModel.sample` produces a schedule by an
independent random walk per agent: at each step it draws a Bernoulli(½)
increment and caps it so the running cumulative does not exceed ``Δ``.
This is one simple way to draw a feasible schedule; it is *not* uniform
over the polytope of feasible schedules, but it suffices for empirical
sanity checking.

Adversary. :meth:`BoundedDelayModel.worst_case_against` is the greedy
adversary used in the bounded-delay certifier: for every vertex
conflict ``(i, j, v, t_i, t_j)`` with ``t_i ≤ t_j`` it tries to advance
agent ``i`` so that ``i`` is still at ``v`` when ``j`` arrives.

Examples
--------
>>> import numpy as np
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> rng = np.random.default_rng(0)
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> sched = BoundedDelayModel(delta=1).sample(Plan.from_paths(a, p), rng)
>>> sum(sched[0]) <= 1
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slackcertify.core.conflict import Conflict, VertexConflict
from slackcertify.core.plan import Plan

__all__ = ["BoundedDelayModel"]


@dataclass(frozen=True, slots=True)
class BoundedDelayModel:
    """Worst-case-bounded per-agent delay model with budget ``delta``.

    Parameters
    ----------
    delta
        Maximum cumulative per-agent delay, in time-steps. Must be
        non-negative.
    """

    delta: int

    def __post_init__(self) -> None:
        """Validate that ``delta`` is non-negative."""
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")

    def sample(self, plan: Plan, rng: np.random.Generator) -> dict[int, list[int]]:
        """Sample a feasible per-agent delay schedule.

        Each agent's vector has length equal to its path makespan; the
        cumulative sum is guaranteed to be ``<= delta`` at every prefix.

        Examples
        --------
        >>> import numpy as np
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> rng = np.random.default_rng(123)
        >>> a = [Agent(id=0, start=(0, 0), goal=(3, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0)])]
        >>> sched = BoundedDelayModel(delta=2).sample(Plan.from_paths(a, p), rng)
        >>> len(sched[0]) == 3 and sum(sched[0]) <= 2
        True
        """
        out: dict[int, list[int]] = {}
        for path in plan.paths:
            steps = path.makespan
            schedule: list[int] = []
            cumulative = 0
            for _ in range(steps):
                if cumulative >= self.delta:
                    schedule.append(0)
                    continue
                inc = int(rng.integers(low=0, high=2))  # Bernoulli(1/2) over {0, 1}
                if cumulative + inc > self.delta:
                    inc = self.delta - cumulative
                schedule.append(inc)
                cumulative += inc
            out[path.agent_id] = schedule
        return out

    def worst_case_against(self, plan: Plan, conflicts: list[Conflict]) -> dict[int, list[int]]:
        """Return a greedy adversarial schedule given a list of conflicts.

        The schedule is initialised to all-zeros. For every
        :class:`VertexConflict` the adversary tries to push the
        *earlier* visitor's effective arrival up to the *later*
        visitor's time:

        * ``gap = t_late - t_early``,
        * if ``cumulative[earlier] + gap <= delta``, ``gap`` units of
          delay are appended at step ``max(0, t_early - 1)``.

        Edge conflicts are ignored by the greedy heuristic; the
        certifier is responsible for handling them via wait insertion.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> from slackcertify.core.conflict import detect_conflicts
        >>> a = [Agent(id=0, start=(0, 0), goal=(3, 0)),
        ...      Agent(id=1, start=(2, 3), goal=(2, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0)]),
        ...      Path(agent_id=1, vertices=[(2, 3), (2, 2), (2, 1), (2, 0)])]
        >>> plan = Plan.from_paths(a, p)
        >>> conflicts = detect_conflicts(plan, delta=2)
        >>> sched = BoundedDelayModel(delta=2).worst_case_against(plan, conflicts)
        >>> all(sum(v) <= 2 for v in sched.values())
        True
        """
        path_steps = {p.agent_id: p.makespan for p in plan.paths}
        schedule: dict[int, list[int]] = {a.id: [0] * path_steps[a.id] for a in plan.agents}
        cumulative: dict[int, int] = {a.id: 0 for a in plan.agents}

        for c in conflicts:
            if not isinstance(c, VertexConflict):
                continue
            if c.t_i <= c.t_j:
                early_agent, t_early, t_late = c.agent_i, c.t_i, c.t_j
            else:
                early_agent, t_early, t_late = c.agent_j, c.t_j, c.t_i
            gap = t_late - t_early
            if gap <= 0:
                continue  # already simultaneous; nothing to advance
            steps = path_steps[early_agent]
            if steps == 0:
                continue
            slot = max(0, t_early - 1)
            if slot >= steps:
                slot = steps - 1
            available = self.delta - cumulative[early_agent]
            advance = min(gap, available)
            if advance <= 0:
                continue
            schedule[early_agent][slot] += advance
            cumulative[early_agent] += advance
        return schedule
