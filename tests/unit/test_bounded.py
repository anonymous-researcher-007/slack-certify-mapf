"""Unit tests for slackcertify.delay.bounded."""

from __future__ import annotations

import numpy as np
import pytest

from slackcertify.core.conflict import detect_conflicts
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.bounded import BoundedDelayModel


def _two_agent_delta_plan() -> Plan:
    """Two agents whose paths cross (2, 0) at t=1 and t=3 — same fixture
    as test_conflict.test_delta_widens_window."""
    agents = [
        Agent(id=0, start=(1, 0), goal=(3, 0)),
        Agent(id=1, start=(2, 3), goal=(2, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(1, 0), (2, 0), (3, 0)]),
        Path(agent_id=1, vertices=[(2, 3), (2, 2), (2, 1), (2, 0)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_negative_delta_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BoundedDelayModel(delta=-1)


@pytest.mark.unit
def test_sample_respects_cumulative_bound() -> None:
    plan = _two_agent_delta_plan()
    rng = np.random.default_rng(0)
    model = BoundedDelayModel(delta=2)
    for _ in range(50):
        sched = model.sample(plan, rng)
        for agent_id, deltas in sched.items():
            assert all(d >= 0 for d in deltas)
            assert sum(deltas) <= 2, f"agent {agent_id} blew the budget: {deltas}"


@pytest.mark.unit
def test_sample_lengths_match_path_makespans() -> None:
    plan = _two_agent_delta_plan()
    rng = np.random.default_rng(1)
    sched = BoundedDelayModel(delta=2).sample(plan, rng)
    expected = {p.agent_id: p.makespan for p in plan.paths}
    assert {k: len(v) for k, v in sched.items()} == expected


@pytest.mark.unit
def test_sample_reproducibility() -> None:
    plan = _two_agent_delta_plan()
    model = BoundedDelayModel(delta=3)
    a = model.sample(plan, np.random.default_rng(42))
    b = model.sample(plan, np.random.default_rng(42))
    assert a == b


@pytest.mark.unit
def test_sample_zero_delta_is_all_zeros() -> None:
    plan = _two_agent_delta_plan()
    rng = np.random.default_rng(7)
    sched = BoundedDelayModel(delta=0).sample(plan, rng)
    assert all(d == 0 for v in sched.values() for d in v)


@pytest.mark.unit
def test_worst_case_against_advances_earlier_agent() -> None:
    plan = _two_agent_delta_plan()
    conflicts = detect_conflicts(plan, delta=2)
    sched = BoundedDelayModel(delta=2).worst_case_against(plan, conflicts)
    # Agent 0 visits (2, 0) at t=1, agent 1 at t=3 -> gap = 2.
    # So agent 0 should accumulate 2 units of delay.
    assert sum(sched[0]) == 2
    # Agent 1 has no earlier conflicts to advance into, stays at zero.
    assert sum(sched[1]) == 0


@pytest.mark.unit
def test_worst_case_respects_delta_budget() -> None:
    plan = _two_agent_delta_plan()
    conflicts = detect_conflicts(plan, delta=2)
    sched = BoundedDelayModel(delta=1).worst_case_against(plan, conflicts)
    # Budget is only 1, so agent 0 can absorb at most 1 unit.
    assert sum(sched[0]) <= 1
