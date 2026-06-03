"""Slack-Certify MAPF — public API.

This module re-exports the user-facing types of the package. Heavyweight
internals (solvers, IO backends, the certifier implementation) live in their
own submodules and should be imported from there.

Examples
--------
>>> from slackcertify import Plan, Path, Agent
>>> a = Agent(id=0, start=(0, 0), goal=(0, 0))
>>> p = Path(agent_id=0, vertices=[(0, 0)])
>>> Plan.from_paths([a], [p]).makespan
0
"""

from __future__ import annotations

from slackcertify._version import __version__
from slackcertify.certify.certificate import Certificate
from slackcertify.core.conflict import Conflict
from slackcertify.core.plan import Agent, Path, Plan

__all__ = [
    "Agent",
    "Certificate",
    "Conflict",
    "Path",
    "Plan",
    "SlackCertify",
    "__version__",
]


class SlackCertify:  # pragma: no cover - placeholder until the certifier facade lands
    """Placeholder for the top-level slack-certifier facade.

    Implemented in :mod:`slackcertify.certify` in a forthcoming change.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Placeholder constructor; raises :class:`NotImplementedError`."""
        raise NotImplementedError(
            "SlackCertify is a placeholder; the orchestration facade will be added in "
            "a follow-up commit (the underlying certifier helpers are already exposed "
            "in slackcertify.certify)."
        )
