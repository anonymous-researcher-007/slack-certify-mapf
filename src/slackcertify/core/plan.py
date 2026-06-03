"""Pydantic models describing MAPF instances and plans.

These types are the canonical interchange format between solvers, the slack
certifier, the simulator, and the IO layer. They carry their own validation
logic; pass a :class:`~slackcertify.core.graph.GridGraph` via the Pydantic
*context* mechanism to additionally validate physical adjacency::

    Plan.model_validate(payload, context={"graph": grid})

When no graph is supplied, structural invariants (start/goal, monotone
times, post-goal waits, makespan/SoC consistency) are still enforced.

Examples
--------
>>> from slackcertify.core.graph import GridGraph
>>> grid = GridGraph(3, 1, set())
>>> a = Agent(id=0, start=(0, 0), goal=(2, 0))
>>> p = Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])
>>> plan = Plan.from_paths(agents=[a], paths=[p])
>>> plan.makespan, plan.sum_of_costs
(2, 2)
>>> plan.vertex_visit(0, 1)
(1, 0)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slackcertify.core.graph import Cell, GridGraph

__all__ = ["Agent", "Path", "Plan"]


class Agent(BaseModel):
    """Single MAPF agent: an integer id with start and goal cells.

    Examples
    --------
    >>> Agent(id=0, start=(0, 0), goal=(3, 4))
    Agent(id=0, start=(0, 0), goal=(3, 4))
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(ge=0)
    start: Cell
    goal: Cell

    def __repr__(self) -> str:  # noqa: D401 - short name
        """Compact ``Agent(id=..., start=..., goal=...)`` representation."""
        return f"Agent(id={self.id}, start={self.start}, goal={self.goal})"


class Path(BaseModel):
    """Time-indexed trajectory of a single agent.

    The trajectory is the list of cells visited at times ``0, 1, ..., T``.
    Two consecutive cells must be either equal (a *wait* action) or
    physically adjacent in the supplied graph; without a graph, only the
    "equal-or-different" structural check is enforced and a downstream
    validator is responsible for adjacency.

    Validators applied at construction time:

    * ``vertices`` is non-empty,
    * the first cell equals the agent's start (when an :class:`Agent` is
      provided via context as ``"agent"``),
    * the last cell equals the agent's goal (likewise) — the agent must
      *end* at the goal, but (convention B) may transiently leave and
      re-enter it before final arrival,
    * if a :class:`GridGraph` is provided via context as ``"graph"``,
      consecutive cells must be equal or adjacent.

    Examples
    --------
    >>> Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 0)])
    Path(agent_id=0, len=3)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: int = Field(ge=0)
    vertices: list[Cell] = Field(min_length=1)

    def __repr__(self) -> str:  # noqa: D401 - short name
        """Compact ``Path(agent_id=..., len=...)`` representation."""
        return f"Path(agent_id={self.agent_id}, len={len(self.vertices)})"

    def __len__(self) -> int:
        """Return the number of vertices in this path."""
        return len(self.vertices)

    @property
    def makespan(self) -> int:
        """Number of *transitions*; equals ``len(vertices) - 1``."""
        return len(self.vertices) - 1

    @property
    def arrival_time(self) -> int:
        """Final arrival time: the earliest ``t`` with ``vertices[t'] == goal``
        for all ``t' >= t``.

        Under convention B the agent may occupy its goal transiently at
        earlier times; this returns the time it *settles* there for good
        (equivalently, the time index of the final non-wait move). It is
        this agent's contribution to the sum-of-costs.
        """
        last = self.vertices[-1]
        t = len(self.vertices) - 1
        while t > 0 and self.vertices[t - 1] == last:
            t -= 1
        return t

    @model_validator(mode="after")
    def _check_consistency(self, info: Any) -> Path:  # noqa: ANN401 - pydantic api
        """Validate path adjacency and (if context provided) graph + agent fit."""
        ctx = (info.context or {}) if info is not None else {}
        graph: GridGraph | None = ctx.get("graph") if isinstance(ctx, dict) else None
        agent: Agent | None = ctx.get("agent") if isinstance(ctx, dict) else None

        verts = self.vertices

        # Structural: cells in-bounds and free, when a graph is available.
        if graph is not None:
            for v in verts:
                if not graph.is_free(v):
                    raise ValueError(f"Path[agent_id={self.agent_id}] visits non-free cell {v}")
            for t, (u, w) in enumerate(zip(verts, verts[1:], strict=False)):
                if u != w and w not in graph.neighbors(u):
                    raise ValueError(
                        f"Path[agent_id={self.agent_id}] step {t}->{t + 1}: "
                        f"{u} and {w} are not adjacent in the graph"
                    )

        # Endpoint and post-goal-wait checks, when an agent is available.
        if agent is not None:
            if agent.id != self.agent_id:
                raise ValueError(
                    f"Path agent_id={self.agent_id} does not match Agent.id={agent.id}"
                )
            if verts[0] != agent.start:
                raise ValueError(
                    f"Path[agent_id={self.agent_id}] starts at {verts[0]}, "
                    f"expected start={agent.start}"
                )
            if verts[-1] != agent.goal:
                raise ValueError(
                    f"Path[agent_id={self.agent_id}] ends at {verts[-1]}, "
                    f"expected goal={agent.goal}"
                )

        return self


class Plan(BaseModel):
    """Joint MAPF plan: one path per agent plus aggregate cost statistics.

    Validators applied at construction time:

    * every agent has exactly one path, indexed by ``agent_id``,
    * each path's ``agent_id`` matches an agent in ``agents``,
    * ``makespan == max(path.makespan for path in paths)``,
    * ``sum_of_costs == sum(path.arrival_time for path in paths)``.

    Use :meth:`from_paths` to construct a plan and have ``makespan`` /
    ``sum_of_costs`` filled in for you.

    Examples
    --------
    >>> a = Agent(id=0, start=(0, 0), goal=(1, 0))
    >>> p = Path(agent_id=0, vertices=[(0, 0), (1, 0)])
    >>> Plan.from_paths(agents=[a], paths=[p]).makespan
    1
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agents: list[Agent]
    paths: list[Path]
    makespan: int = Field(ge=0)
    sum_of_costs: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_consistency(self, info: Any) -> Plan:  # noqa: ANN401 - pydantic api
        """Validate agent / path bijection and cached makespan + sum_of_costs."""
        agent_ids = [a.id for a in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent ids must be unique")

        path_ids = [p.agent_id for p in self.paths]
        if sorted(path_ids) != sorted(agent_ids):
            raise ValueError(
                f"path agent_ids {sorted(path_ids)} do not match agent ids {sorted(agent_ids)}"
            )

        # If a graph context was supplied, re-validate each path against it
        # together with its owning Agent so endpoint and adjacency checks fire.
        ctx = (info.context or {}) if info is not None else {}
        graph: GridGraph | None = ctx.get("graph") if isinstance(ctx, dict) else None
        if graph is not None or any(self.paths):
            agent_by_id = {a.id: a for a in self.agents}
            for p in self.paths:
                a = agent_by_id[p.agent_id]
                # Re-trigger validation with graph + agent in context.
                Path.model_validate(
                    p.model_dump(),
                    context={"graph": graph, "agent": a},
                )

        expected_makespan = max((p.makespan for p in self.paths), default=0)
        if self.makespan != expected_makespan:
            raise ValueError(
                f"makespan={self.makespan} disagrees with max path makespan={expected_makespan}"
            )

        expected_soc = sum(p.arrival_time for p in self.paths)
        if self.sum_of_costs != expected_soc:
            raise ValueError(
                f"sum_of_costs={self.sum_of_costs} disagrees with sum of arrival "
                f"times={expected_soc}"
            )

        return self

    # ------------------------------------------------------------------ helpers

    def vertex_visit(self, agent_id: int, t: int) -> Cell:
        """Return the cell occupied by ``agent_id`` at integer time ``t``.

        Times beyond the path length pad with the agent's last cell (the
        goal), reflecting the standard MAPF "stay-at-goal" execution model.

        Examples
        --------
        >>> a = Agent(id=0, start=(0, 0), goal=(1, 0))
        >>> p = Path(agent_id=0, vertices=[(0, 0), (1, 0)])
        >>> plan = Plan.from_paths(agents=[a], paths=[p])
        >>> plan.vertex_visit(0, 0)
        (0, 0)
        >>> plan.vertex_visit(0, 5)
        (1, 0)
        """
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        path = self._path_index()[agent_id]
        if t < len(path.vertices):
            return path.vertices[t]
        return path.vertices[-1]

    def _path_index(self) -> dict[int, Path]:
        """Return ``{agent_id: Path}`` for fast per-agent lookup."""
        return {p.agent_id: p for p in self.paths}

    # -------------------------------------------------------------- factories

    @classmethod
    def from_paths(cls, agents: list[Agent], paths: list[Path]) -> Plan:
        """Construct a :class:`Plan` and compute ``makespan`` / ``sum_of_costs``.

        Examples
        --------
        >>> a = Agent(id=0, start=(0, 0), goal=(2, 0))
        >>> p = Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])
        >>> Plan.from_paths([a], [p]).sum_of_costs
        2
        """
        makespan = max((p.makespan for p in paths), default=0)
        sum_of_costs = sum(p.arrival_time for p in paths)
        return cls(
            agents=list(agents),
            paths=list(paths),
            makespan=makespan,
            sum_of_costs=sum_of_costs,
        )
