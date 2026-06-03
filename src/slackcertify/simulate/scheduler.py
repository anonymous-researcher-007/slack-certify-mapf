"""TPG-driven scheduler used by the deterministic executor.

A :class:`TPGScheduler` is a thin wrapper around a temporal-plan graph
that answers the single question "given the current per-agent execution
state, may agent ``i`` advance to its next path step?" — by checking
that every TPG predecessor of agent ``i``'s next-node has already been
visited by its owning agent.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> from slackcertify.core.tpg import build_tpg
>>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
>>> plan = Plan.from_paths(a, p)
>>> sched = TPGScheduler(plan, build_tpg(plan))
>>> sched.next_action(0, {0: 0})
'advance'
>>> sched.next_action(0, {0: 1})  # already at goal
'wait'
"""

from __future__ import annotations

from typing import Literal

from slackcertify.core.plan import Plan
from slackcertify.core.tpg import TPG, TPGNode

__all__ = ["TPGScheduler"]

Action = Literal["advance", "wait"]


class TPGScheduler:
    """Per-step "advance or wait" decision oracle backed by a TPG.

    Parameters
    ----------
    plan
        The plan being executed. Used to look up each agent's path.
    tpg
        The plan's temporal plan graph. Type-1, Type-2 and any repair
        edges are honoured.
    """

    def __init__(self, plan: Plan, tpg: TPG) -> None:
        """Store ``plan`` + its derived ``tpg`` for per-agent action lookups."""
        self._plan: Plan = plan
        self._tpg: TPG = tpg
        self._paths: dict[int, list[tuple[int, int]]] = {
            p.agent_id: list(p.vertices) for p in plan.paths
        }

    def next_action(self, agent_id: int, current_state: dict[int, int]) -> Action:
        """Return ``'advance'`` if ``agent_id`` may step forward, else ``'wait'``.

        Parameters
        ----------
        agent_id
            The agent being queried.
        current_state
            Mapping from each agent id to its current path index
            (i.e. the index of its current vertex in its path).

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> from slackcertify.core.tpg import build_tpg
        >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0)),
        ...      Agent(id=1, start=(2, 0), goal=(0, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
        ...      Path(agent_id=1, vertices=[(2, 0), (1, 0), (0, 0)])]
        >>> plan = Plan.from_paths(a, p)
        >>> tpg = build_tpg(plan)
        >>> sched = TPGScheduler(plan, tpg)
        >>> sched.next_action(0, {0: 0, 1: 0})
        'advance'
        """
        if agent_id not in self._paths:
            raise KeyError(f"unknown agent_id={agent_id}")
        verts = self._paths[agent_id]
        idx = current_state.get(agent_id, 0)
        if idx >= len(verts) - 1:
            return "wait"
        next_node = TPGNode(agent_id, verts[idx + 1], idx + 1)
        if next_node not in self._tpg:
            return "wait"
        for pred in self._tpg.graph.predecessors(next_node):
            other_idx = current_state.get(pred.agent_id, 0)
            if other_idx < pred.time_step:
                return "wait"
        return "advance"

    @property
    def plan(self) -> Plan:
        """Return the plan this scheduler was built for."""
        return self._plan

    @property
    def tpg(self) -> TPG:
        """Return the underlying TPG."""
        return self._tpg
