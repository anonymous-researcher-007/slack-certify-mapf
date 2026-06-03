"""Solver registry — name → class lookup."""

from __future__ import annotations

from typing import Any

from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.eecbs import EECBSSolver
from slackcertify.solvers.lacam import LaCAMStarSolver
from slackcertify.solvers.pibt import PIBTSolver

__all__ = ["SOLVERS", "get_solver"]


SOLVERS: dict[str, type[Any]] = {
    "eecbs": EECBSSolver,
    "lacam_star": LaCAMStarSolver,
    "pibt": PIBTSolver,
    "fake": FakeSolver,
}


def get_solver(name: str, **kwargs: Any) -> Any:  # noqa: ANN401 - heterogeneous solver kwargs
    """Construct the solver registered under ``name``.

    Examples
    --------
    >>> get_solver("fake").NAME
    'fake'
    >>> get_solver("eecbs").NAME
    'eecbs'
    """
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; choose from {sorted(SOLVERS)}")
    return SOLVERS[name](**kwargs)
