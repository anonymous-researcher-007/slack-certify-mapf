"""Solver wrappers (subprocess-based) and the in-process FakeSolver."""

from __future__ import annotations

from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.base import (
    BinarySolverBase,
    SolverError,
    SolverNotFoundError,
    SolverOutputParseError,
    SolverProtocol,
    SolverTimeoutError,
)
from slackcertify.solvers.eecbs import EECBSSolver
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.solvers.pibt import PIBTSolver
from slackcertify.solvers.registry import SOLVERS, get_solver

__all__ = [
    "BinarySolverBase",
    "EECBSSolver",
    "FakeSolver",
    "LaCAMStarSolver",
    "PIBTSolver",
    "SOLVERS",
    "SolverError",
    "SolverNotFoundError",
    "SolverOutputParseError",
    "SolverProtocol",
    "SolverTimeoutError",
    "get_solver",
]
