"""Slack-certifier driver and helpers (Algorithm 1).

Re-exports the entry point :func:`slack_certify` and the module-level
exceptions used to signal failure modes.
"""

from __future__ import annotations

from slackcertify.repair.algorithm import (
    BudgetInfeasibleError,
    CertificationFailure,
    WaitInfeasibleError,
    slack_certify,
)
from slackcertify.repair.deadlock import DeadlockError
from slackcertify.repair.wait_feasibility import is_wait_resolvable

__all__ = [
    "BudgetInfeasibleError",
    "CertificationFailure",
    "DeadlockError",
    "WaitInfeasibleError",
    "is_wait_resolvable",
    "slack_certify",
]
