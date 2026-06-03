"""Regression tests for wait-infeasibility detection (PART A precheck).

A head-on swap on a 1-D corridor cannot be resolved by waits alone: both
agents traverse exactly the same cell set in opposite directions, so no
amount of waits at any visit creates a side-bay to step aside into. The
PART-A precheck in :func:`slack_certify` and
:func:`solve_optimal_wait_insertion` must catch this upfront with an
actionable :class:`WaitInfeasibleError`, not with a generic "did not
converge" or — worse — a silent false-optimum.
"""

from __future__ import annotations

import pytest

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.repair.algorithm import WaitInfeasibleError, slack_certify


def _make_corridor_swap(n: int) -> tuple[GridGraph, Plan]:
    """Construct an n-cell 1-D corridor with two agents swapping ends.

    The grid is sized to contain the paths (``width=1, height=n``); the
    paths themselves use vertical coordinates ``(0, 0)..(0, n-1)``.
    """
    graph = GridGraph(width=1, height=n, obstacles=set(), connectivity=4)
    agents = [
        Agent(id=0, start=(0, 0), goal=(0, n - 1)),
        Agent(id=1, start=(0, n - 1), goal=(0, 0)),
    ]
    path_0 = Path(agent_id=0, vertices=[(0, i) for i in range(n)])
    path_1 = Path(agent_id=1, vertices=[(0, n - 1 - i) for i in range(n)])
    return graph, Plan.from_paths(agents, [path_0, path_1])


@pytest.mark.property
@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_corridor_swap_raises_wait_infeasible(n: int) -> None:
    _, plan = _make_corridor_swap(n)
    with pytest.raises(WaitInfeasibleError) as excinfo:
        slack_certify(plan, mode="bounded", delta=1)
    assert len(excinfo.value.unresolvable_conflicts) > 0
    # The message must explicitly mention the structural-infeasibility
    # reason; "did not converge" would mean the precheck was bypassed.
    assert "structural" in str(excinfo.value) or "wait insertion cannot" in str(excinfo.value)
