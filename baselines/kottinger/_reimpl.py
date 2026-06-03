"""Kottinger 2024 baseline: explicit fail-loud stub.

Earlier versions of this module shipped a "sandbox fallback" that
delegated to :func:`slackcertify.repair.slack_certify` when the
upstream binary was unavailable. That fallback was structurally
meaningless as a Kottinger comparison — it produced rows tagged
``_reimpl`` that looked like Kottinger results but were actually our
own algorithm's output, and a reviewer (or future maintainer) might
forget to filter them out of paper-grade plots and tables.

The audit's D3 resolution removes the silent fallback. Calling
:func:`kottinger_offline_solve` from this module now raises
:class:`SolverNotFoundError` immediately, surfacing the missing
upstream binary loudly so the §V runner records
``status="binary_missing"`` per the established convention.

The genuine implementation of Kottinger's algorithm is the
upstream binary at ``third_party/delay-introduction``
(``aria-systems-group/Delay-Robust-MAPF``). See
``scripts/install_baselines.sh kottinger`` for the build step
and ``third_party/README.md`` for the upstream URL and the
``TODO_VERIFY`` placeholders that must be confirmed against the
upstream README on the first real build.

Online mode is not on the §V critical path and raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

from baselines._paths import external_bin
from slackcertify.core.plan import Plan
from slackcertify.solvers.base import SolverNotFoundError

__all__ = [
    "KOTTINGER_BIN_PATH",
    "kottinger_offline_solve",
    "kottinger_online_solve",
]


KOTTINGER_BIN_PATH = external_bin("kottinger")


def kottinger_offline_solve(plan: Plan, delta: int, time_limit_s: float) -> Plan:
    """Always raise :class:`SolverNotFoundError`; the upstream binary is required.

    See the module docstring for the rationale: a silent fallback to
    ``slack_certify`` tagged the output as ``_reimpl`` and risked
    contaminating §V comparisons with our own algorithm's results.

    Parameters
    ----------
    plan, delta, time_limit_s
        Ignored; accepted for interface parity with the upstream
        binary's signature so callers can swap the two at the same
        call site without code changes.

    Raises
    ------
    SolverNotFoundError
        Always. The error message points at
        ``scripts/install_baselines.sh kottinger`` and the upstream
        URL.
    """
    _ = (plan, delta, time_limit_s)  # accepted for interface parity
    raise SolverNotFoundError(
        "Kottinger 2024 baseline requires the upstream binary at "
        f"{KOTTINGER_BIN_PATH}. Build via "
        "`bash scripts/install_baselines.sh kottinger`. See "
        "third_party/README.md for the upstream URL and TODO_VERIFY "
        "placeholders that must be confirmed against "
        "aria-systems-group/Delay-Robust-MAPF's README on first build."
    )


def kottinger_online_solve(plan: Plan, observed_delay: object, time_limit_s: float) -> Plan:  # noqa: ARG001
    """Reactive single-event Kottinger solve.

    Not on the §V critical path; included for completeness only.

    Raises
    ------
    NotImplementedError
        Always — implement this when a §VI ablation requires it.
    """
    raise NotImplementedError(
        "kottinger_online_solve is not on the §V critical path and is not "
        "implemented yet. Add it when an ablation requires reactive solves."
    )
