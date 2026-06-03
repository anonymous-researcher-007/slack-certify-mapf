"""Discrete-time joint executor for slack-certified plans under delays.

The :class:`Executor` walks wall-clock time tick-by-tick. At each tick it
asks every agent's ``(current_path_index, remaining_wait)`` state to
either consume one wait tick or advance to the next path vertex (and
load the new vertex's prescribed delay). Vertex collisions and
head-on-swap edge collisions are detected and recorded as they happen.

Example
-------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> trace = Executor(Plan.from_paths(a, p), {0: [0, 0]}).run()
>>> trace.success and trace.collisions == []
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slackcertify.core.plan import Plan

__all__ = ["ExecutionTrace", "Executor"]

_Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    """Recorded collision."""

    wall_t: int
    kind: str  # "vertex" or "edge"
    agent_i: int
    agent_j: int
    where: tuple[_Cell, _Cell] | _Cell


@dataclass(slots=True)
class ExecutionTrace:
    """Result of executing a plan under a delay schedule.

    Attributes
    ----------
    success
        ``True`` iff no collision was recorded.
    collisions
        List of :class:`CollisionEvent`, in chronological order.
    final_t
        Wall-clock tick at which every agent first reached its goal.
    realised_arrival
        ``{agent_id: wall_clock_tick}`` of each agent's first arrival.
    """

    success: bool
    collisions: list[CollisionEvent] = field(default_factory=list)
    final_t: int = 0
    realised_arrival: dict[int, int] = field(default_factory=dict)


class Executor:
    """Joint discrete-time MAPF executor under a fixed per-agent delay schedule.

    Parameters
    ----------
    plan
        The plan to execute.
    delay_schedule
        Per-agent integer delay vector (length equals path makespan).
        Missing entries are treated as zero delays. Each entry is the
        number of *extra* ticks the agent waits at the corresponding
        path vertex before moving on.
    """

    def __init__(self, plan: Plan, delay_schedule: dict[int, list[int]]) -> None:
        """Build per-agent state from ``plan`` and the per-step delay vector."""
        self._plan: Plan = plan
        self._delays: dict[int, list[int]] = {
            p.agent_id: list(delay_schedule.get(p.agent_id, [0] * p.makespan)) for p in plan.paths
        }
        self._paths: dict[int, list[_Cell]] = {p.agent_id: list(p.vertices) for p in plan.paths}
        # Per-agent (current path index, remaining wait at this vertex).
        self._state: dict[int, tuple[int, int]] = {
            aid: (0, self._initial_wait(aid)) for aid in self._paths
        }
        self._t: int = 0
        self._collisions: list[CollisionEvent] = []
        self._arrival: dict[int, int] = {}
        self._check_collisions(prev_positions=None)

    # ----------------------------------------------------------- helpers

    def _initial_wait(self, aid: int) -> int:
        """Return the delay prescribed at agent ``aid``'s starting vertex."""
        ds = self._delays[aid]
        return ds[0] if ds else 0

    def _position(self, aid: int) -> _Cell:
        """Return the cell currently occupied by agent ``aid``."""
        idx, _ = self._state[aid]
        return self._paths[aid][idx]

    def _at_goal(self, aid: int) -> bool:
        """Return ``True`` iff agent ``aid`` has reached its final path vertex."""
        idx, _ = self._state[aid]
        return idx >= len(self._paths[aid]) - 1

    # ----------------------------------------------------------- API

    @property
    def t(self) -> int:
        """Current wall-clock tick (number of completed :meth:`step` calls)."""
        return self._t

    def state(self) -> dict[int, tuple[int, int]]:
        """Return a shallow copy of the per-agent state."""
        return dict(self._state)

    def step(self) -> dict[int, _Cell]:
        """Advance one wall-clock tick and return the new joint configuration."""
        prev_positions = {aid: self._position(aid) for aid in self._paths}
        new_state: dict[int, tuple[int, int]] = {}
        for aid, (idx, remaining) in self._state.items():
            verts = self._paths[aid]
            if remaining > 0:
                new_state[aid] = (idx, remaining - 1)
            elif idx < len(verts) - 1:
                new_idx = idx + 1
                # Wait for the delay prescribed at the *new* vertex (i.e. the
                # delay the schedule applies to the move that lands on new_idx).
                ds = self._delays[aid]
                d_new = ds[new_idx] if new_idx < len(ds) else 0
                new_state[aid] = (new_idx, d_new)
                # First arrival at the goal is recorded the moment the agent
                # reaches it.
                if new_idx == len(verts) - 1 and aid not in self._arrival:
                    self._arrival[aid] = self._t + 1
            else:
                new_state[aid] = (idx, 0)
                self._arrival.setdefault(aid, self._t)
        self._state = new_state
        self._t += 1
        self._check_collisions(prev_positions=prev_positions)
        return {aid: self._position(aid) for aid in self._paths}

    def collisions(self) -> list[CollisionEvent]:
        """Return a copy of the collisions recorded so far."""
        return list(self._collisions)

    def is_done(self) -> bool:
        """Return ``True`` once every agent is at its goal with zero remaining wait."""
        return all(self._at_goal(aid) and self._state[aid][1] == 0 for aid in self._paths)

    def run(self, max_ticks: int | None = None) -> ExecutionTrace:
        """Step until every agent has arrived (or ``max_ticks`` is hit)."""
        if max_ticks is None:
            max_ticks = (
                sum(p.makespan + sum(self._delays[p.agent_id]) for p in self._plan.paths) + 16
            )
        while not self.is_done() and self._t < max_ticks:
            self.step()
        for aid in self._paths:
            self._arrival.setdefault(aid, self._t)
        return ExecutionTrace(
            success=not self._collisions,
            collisions=list(self._collisions),
            final_t=self._t,
            realised_arrival=dict(self._arrival),
        )

    # ----------------------------------------------------------- internals

    def _check_collisions(self, prev_positions: dict[int, _Cell] | None) -> None:
        """Record vertex / edge collisions at the current tick onto ``_collisions``."""
        positions: dict[int, _Cell] = {aid: self._position(aid) for aid in self._paths}
        # Vertex
        cells: dict[_Cell, int] = {}
        for aid, v in positions.items():
            if v in cells:
                self._collisions.append(
                    CollisionEvent(
                        wall_t=self._t,
                        kind="vertex",
                        agent_i=cells[v],
                        agent_j=aid,
                        where=v,
                    )
                )
            else:
                cells[v] = aid
        # Edge (head-on swap) requires the previous tick's positions.
        if prev_positions is not None:
            agent_ids = list(positions)
            for i_idx, a in enumerate(agent_ids):
                for b in agent_ids[i_idx + 1 :]:
                    pa, pb = prev_positions[a], prev_positions[b]
                    na, nb = positions[a], positions[b]
                    if pa != na and pb != nb and pa == nb and pb == na:
                        self._collisions.append(
                            CollisionEvent(
                                wall_t=self._t,
                                kind="edge",
                                agent_i=a,
                                agent_j=b,
                                where=(pa, pb),
                            )
                        )
