"""Detect which ILP backends are usable on the current machine.

The slack-certify-mapf ILP layer wraps three backends through PuLP:

* **HiGHS** — the default, open-source MIP solver bundled via the
  ``highspy`` Python package.
* **CBC** — the COIN-OR solver, shipped with PuLP itself; reliable
  fallback when HiGHS is unavailable.
* **Gurobi** — commercial, gated by the presence of either
  ``gurobi_cl`` on ``PATH`` or the ``GUROBI_HOME`` environment
  variable.

These helpers are deliberately import-tolerant: if PuLP itself is
missing, :func:`available_solvers` returns the empty list rather
than raising, so callers can pick a graceful failure mode.
"""

from __future__ import annotations

import os
import shutil

__all__ = ["available_solvers", "default_solver"]


def _has_highs() -> bool:
    """Return ``True`` iff the HiGHS MIP backend is invokable through pulp."""
    try:
        import pulp  # noqa: PLC0415 - lazy
    except ImportError:
        return False
    if hasattr(pulp, "HiGHS"):
        try:
            if pulp.HiGHS().available():
                return True
        except Exception:
            pass
    if hasattr(pulp, "HiGHS_CMD"):
        try:
            if pulp.HiGHS_CMD().available():
                return True
        except Exception:
            pass
    return False


def _has_cbc() -> bool:
    """Return ``True`` iff the COIN-OR CBC backend is invokable through pulp."""
    try:
        import pulp  # noqa: PLC0415 - lazy
    except ImportError:
        return False
    try:
        return bool(pulp.PULP_CBC_CMD().available())
    except Exception:
        return False


def _has_gurobi() -> bool:
    """Return ``True`` iff the Gurobi installation is locatable on the system."""
    if shutil.which("gurobi_cl") is not None:
        return True
    return bool(os.environ.get("GUROBI_HOME"))


def available_solvers() -> list[str]:
    """Return the names of MIP backends invokable on this machine.

    Order: ``highs`` first (preferred), then ``cbc``, then ``gurobi``.

    Examples
    --------
    >>> isinstance(available_solvers(), list)
    True
    """
    out: list[str] = []
    if _has_highs():
        out.append("highs")
    if _has_cbc():
        out.append("cbc")
    if _has_gurobi():
        out.append("gurobi")
    return out


def default_solver() -> str:
    """Return the first available backend or raise :class:`RuntimeError`.

    Examples
    --------
    >>> default_solver() in {"highs", "cbc", "gurobi"}
    True
    """
    av = available_solvers()
    if not av:
        raise RuntimeError("no ILP solver available; install with: pip install 'slackcertify[ilp]'")
    return av[0]
