"""Heuristic per-tick adversarial delay protocol (§V P1).

At each simulator tick the adversary chooses one agent (among those
whose cumulative delay is still strictly below ``delta``) to stall —
specifically the agent whose +1 stall most decreases the wall-clock
arrival gap of an unresolved unsafe pair, tie-broken by the smallest
current gap (i.e., the pair closest to colliding). Because a single
stall changes any one pair's gap by exactly 1, the maximum-reduction
criterion ties across every attackable pair; the smallest-gap tie-break
is therefore the effective primary selector.

The :class:`DelayModel` protocol expects an offline schedule
``dict[agent_id, list[int]]``. Since the plan's path indices and the
mapping from path index to nominal arrival time are fixed, we can
simulate the per-tick decision inside :meth:`sample` and emit the
resulting schedule; the :class:`~slackcertify.simulate.executor.Executor`
will replay it tick-for-tick.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> import numpy as np
>>> HeuristicAdversaryDelay(delta=1).sample(Plan.from_paths(a, p), np.random.default_rng(0))
{0: [0, 0]}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slackcertify.certify.bounded import unsafe_pairs_bounded
from slackcertify.core.conflict import VertexConflict
from slackcertify.core.plan import Plan

__all__ = ["HeuristicAdversaryDelay"]


@dataclass(frozen=True, slots=True)
class HeuristicAdversaryDelay:
    """Heuristic per-tick gap-collapsing adversary under cumulative budget ``delta``.

    Parameters
    ----------
    delta
        Per-agent cumulative-delay budget. The adversary never schedules
        an agent past ``D_i(t) = delta``.
    """

    delta: int

    def __post_init__(self) -> None:
        """Validate ``delta`` is non-negative."""
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")

    def sample(
        self,
        plan: Plan,
        rng: np.random.Generator,  # noqa: ARG002 - protocol parity; adversary is deterministic
    ) -> dict[int, list[int]]:
        """Build a per-agent delay schedule by simulating the adversary tick-by-tick.

        The schedule format matches every other model in
        :mod:`slackcertify.delay`: ``out[aid][k]`` is the extra wait
        agent ``aid`` performs at path-vertex index ``k``.
        """
        path_makespan: dict[int, int] = {p.agent_id: p.makespan for p in plan.paths}
        delay: dict[int, list[int]] = {
            aid: [0] * mk for aid, mk in path_makespan.items()
        }
        if self.delta == 0 or not path_makespan:
            return delay

        # Vertex-conflict pair specs: (agent_i, agent_j, path_idx_i, path_idx_j).
        # The §V protocol is phrased in terms of arrival-gap collapse, which
        # is the vertex-conflict formulation; edge swaps are handled by the
        # repair, not the adversary.
        unsafe = unsafe_pairs_bounded(plan, self.delta)
        pair_specs: list[tuple[int, int, int, int]] = [
            (c.agent_i, c.agent_j, c.t_i, c.t_j)
            for c in unsafe
            if isinstance(c, VertexConflict)
        ]
        if not pair_specs:
            return delay

        cum_delay: dict[int, int] = {aid: 0 for aid in path_makespan}
        cur_idx: dict[int, int] = {aid: 0 for aid in path_makespan}
        finished: dict[int, bool] = {aid: mk == 0 for aid, mk in path_makespan.items()}

        # Bound the simulation: longest path plus the total stall budget.
        max_ticks = max(path_makespan.values()) + sum(
            min(self.delta, mk) for mk in path_makespan.values()
        ) + 1

        for _ in range(max_ticks):
            if all(finished.values()):
                break

            # Build the (gap, agent_to_stall) candidate list for this tick.
            # An agent is "stallable" iff it hasn't reached the goal and
            # hasn't exhausted its budget.
            best_aid: int | None = None
            best_gap: int | None = None
            for i, j, ti, tj in pair_specs:
                # Skip pairs where either agent has already moved past the
                # conflict vertex — the threat is realised (or missed) and
                # further stalls cannot reduce its arrival gap.
                if cur_idx[i] > ti or cur_idx[j] > tj:
                    continue
                arr_i = ti + cum_delay[i]
                arr_j = tj + cum_delay[j]
                if arr_i < arr_j and not finished[i] and cum_delay[i] < self.delta:
                    gap = arr_j - arr_i
                    if best_gap is None or gap < best_gap:
                        best_aid, best_gap = i, gap
                elif arr_j < arr_i and not finished[j] and cum_delay[j] < self.delta:
                    gap = arr_i - arr_j
                    if best_gap is None or gap < best_gap:
                        best_aid, best_gap = j, gap

            # Apply the tick transition. The stalled agent (if any) stays
            # put and accrues +1 delay at its current vertex; every other
            # non-finished agent advances one path index.
            for aid, fin in finished.items():
                if fin:
                    continue
                if aid == best_aid:
                    delay[aid][cur_idx[aid]] += 1
                    cum_delay[aid] += 1
                else:
                    cur_idx[aid] += 1
                    if cur_idx[aid] >= path_makespan[aid]:
                        finished[aid] = True

        return delay
