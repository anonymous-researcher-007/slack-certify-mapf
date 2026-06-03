"""Pure-function plan / certificate validators.

These helpers wrap the per-model validation logic in :mod:`slackcertify.core.plan`
and :mod:`slackcertify.core.conflict` to give callers a single, declarative
entry point for "is this artefact well-formed?" questions. They raise
:class:`ValueError` on the first problem they find, with a message naming the
offending plan element.

Examples
--------
>>> from slackcertify.core.graph import GridGraph
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> g = GridGraph(2, 1, set())
>>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
>>> validate_plan(Plan.from_paths(a, p), g)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from slackcertify.core.conflict import detect_conflicts
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Path, Plan

__all__ = [
    "validate_plan",
    "validate_certificate",
    "assert_conflict_free",
]


@runtime_checkable
class _CertificateLike(Protocol):
    """Minimum interface a certificate must expose for :func:`validate_certificate`.

    The real :class:`slackcertify.Certificate` will satisfy this; the
    placeholder used pre-implementation does not, and the validator
    raises :class:`TypeError` accordingly.
    """

    nodes: Any  # iterable of (agent_id, vertex, time_step)


def validate_plan(plan: Plan, graph: GridGraph) -> None:
    """Re-validate ``plan`` against ``graph``.

    The Pydantic models already enforce structural invariants at
    construction time. This function additionally checks adjacency and
    free-cell membership against the supplied :class:`GridGraph` by
    re-running each :class:`Path` validator with the graph in context.

    Raises
    ------
    ValueError
        On the first invariant violation, with a message that names the
        offending agent / time-step / cell.

    Examples
    --------
    >>> from slackcertify.core.graph import GridGraph
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> g = GridGraph(2, 1, {(1, 0)})
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> validate_plan(Plan.from_paths(a, p), g)
    """
    agent_by_id = {a.id: a for a in plan.agents}
    for path in plan.paths:
        if path.agent_id not in agent_by_id:
            raise ValueError(f"plan path agent_id={path.agent_id} has no matching agent")
        Path.model_validate(
            path.model_dump(),
            context={"graph": graph, "agent": agent_by_id[path.agent_id]},
        )


def validate_certificate(cert: object, plan: Plan) -> None:
    """Check that every node referenced by ``cert`` exists in ``plan``.

    A certificate is treated as a duck-typed object exposing an
    iterable ``nodes`` attribute whose elements are
    ``(agent_id, vertex, time_step)`` triples. For each triple, the
    function asserts that ``plan.vertex_visit(agent_id, time_step)``
    returns the matching ``vertex``.

    Raises
    ------
    TypeError
        If ``cert`` does not expose a ``nodes`` attribute.
    ValueError
        If ``cert`` references an unknown agent, an out-of-range
        time-step, or a cell that disagrees with the plan.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> plan = Plan.from_paths(a, p)
    >>> validate_certificate(SimpleNamespace(nodes=[(0, (0, 0), 0)]), plan)
    """
    if not hasattr(cert, "nodes"):
        raise TypeError(f"validate_certificate: {type(cert).__name__} has no 'nodes' attribute")
    agent_ids = {a.id for a in plan.agents}
    for node in cert.nodes:
        if not (isinstance(node, tuple) and len(node) == 3):
            raise ValueError(
                f"certificate node {node!r} is not a (agent_id, vertex, time_step) triple"
            )
        agent_id, vertex, t = node
        if agent_id not in agent_ids:
            raise ValueError(f"certificate references unknown agent {agent_id}")
        if not isinstance(t, int) or t < 0:
            raise ValueError(f"certificate node {node!r}: time_step must be a non-negative int")
        actual = plan.vertex_visit(agent_id, t)
        if tuple(vertex) != actual:
            raise ValueError(
                f"certificate node {node!r} disagrees with plan: "
                f"plan.vertex_visit({agent_id}, {t}) == {actual}"
            )


def assert_conflict_free(plan: Plan, delta: int = 0) -> None:
    """Raise :class:`ValueError` if ``plan`` has any conflict under tolerance ``delta``.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
    >>> assert_conflict_free(Plan.from_paths(a, p))
    """
    conflicts = detect_conflicts(plan, delta=delta)
    if conflicts:
        first = conflicts[0]
        raise ValueError(
            f"plan has {len(conflicts)} conflict(s) at delta={delta}; " f"first: {first}"
        )
