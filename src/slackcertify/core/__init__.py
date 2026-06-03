"""Core type system for :mod:`slackcertify`.

Re-exports the canonical environment, plan, and conflict primitives so that
downstream modules can simply ``from slackcertify.core import ...``.
"""

from __future__ import annotations

from slackcertify.core.conflict import (
    Conflict,
    EdgeConflict,
    VertexConflict,
    detect_conflicts,
)
from slackcertify.core.graph import Cell, GridGraph
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.core.reservation import ReservationTable
from slackcertify.core.tpg import TPG, TPGNode, build_tpg
from slackcertify.core.validators import (
    assert_conflict_free,
    validate_certificate,
    validate_plan,
)

__all__ = [
    "Agent",
    "Cell",
    "Conflict",
    "EdgeConflict",
    "GridGraph",
    "Path",
    "Plan",
    "ReservationTable",
    "TPG",
    "TPGNode",
    "VertexConflict",
    "assert_conflict_free",
    "build_tpg",
    "detect_conflicts",
    "validate_certificate",
    "validate_plan",
]
