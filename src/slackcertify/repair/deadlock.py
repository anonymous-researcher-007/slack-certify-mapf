"""Deadlock guards and an incremental linear-extension tracker.

Every wait insertion can in principle introduce a Type-2 / repair edge that
closes a cycle in the temporal plan graph. The certifier therefore calls
:func:`check_acyclicity_invariant` after every repair step; if a cycle has
appeared, :class:`DeadlockError` is raised before the (now unsafe) plan is
ever returned to the caller.

:class:`LinearExtensionTracker` materialises the ``t + i / (n + 1)`` linear
extension used in the Lemma 1 proof and refreshes it on demand. The current
implementation re-derives the extension from scratch on each
:meth:`LinearExtensionTracker.update`; a future revision can do so
incrementally.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> from slackcertify.core.tpg import build_tpg
>>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
>>> tpg = build_tpg(Plan.from_paths(a, p))
>>> check_acyclicity_invariant(tpg)
"""

from __future__ import annotations

from slackcertify.core.tpg import TPG, TPGNode

__all__ = ["DeadlockError", "LinearExtensionTracker", "check_acyclicity_invariant"]


class DeadlockError(Exception):
    """The TPG contains a cycle — the wait-insertion induced a deadlock."""


def check_acyclicity_invariant(tpg: TPG) -> None:
    """Raise :class:`DeadlockError` if ``tpg`` is not a DAG.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> from slackcertify.core.tpg import build_tpg
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> check_acyclicity_invariant(build_tpg(Plan.from_paths(a, p)))
    """
    if not tpg.is_acyclic():
        raise DeadlockError(
            "augmented temporal plan graph contains a cycle; the last repair "
            "edge induces a deadlock"
        )


class LinearExtensionTracker:
    """Maintain the ``t + i / (n + 1)`` linear extension across repair edges.

    Parameters
    ----------
    tpg
        The TPG to track. ``update()`` must be called after any
        in-place edit (repair edge insertion, etc.) to refresh the
        cached extension.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> from slackcertify.core.tpg import build_tpg
    >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
    >>> tracker = LinearExtensionTracker(build_tpg(Plan.from_paths(a, p)))
    >>> sigma = tracker.extension()
    >>> all(int(v) == n.time_step for n, v in sigma.items())
    True
    """

    def __init__(self, tpg: TPG) -> None:
        """Snapshot the TPG and compute the initial linear-extension ranks."""
        self._tpg = tpg
        self._sigma: dict[TPGNode, float] = {}
        self.update()

    def update(self) -> None:
        """Recompute the linear extension from the current TPG state."""
        check_acyclicity_invariant(self._tpg)
        self._sigma = self._tpg.linear_extension()

    def extension(self) -> dict[TPGNode, float]:
        """Return the current ``node -> sigma'`` mapping (a fresh copy)."""
        return dict(self._sigma)

    def respects(self, u: TPGNode, v: TPGNode) -> bool:
        """Return ``True`` if ``sigma'(u) < sigma'(v)`` under the current extension."""
        return self._sigma[u] < self._sigma[v]
