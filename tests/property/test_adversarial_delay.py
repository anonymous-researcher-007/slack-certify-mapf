"""Property tests for :class:`HeuristicAdversaryDelay` (§V P1 protocol)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.adversarial import HeuristicAdversaryDelay
from tests.property.strategies import cross_plans

# --------------------------------------------------------------- test 1


@given(plan=cross_plans(), delta=st.integers(min_value=1, max_value=3))
@settings(max_examples=40, deadline=None)
def test_adversarial_respects_budget(plan: Plan, delta: int) -> None:
    """No agent's cumulative delay should ever exceed ``delta``."""
    schedule = HeuristicAdversaryDelay(delta=delta).sample(plan, np.random.default_rng(0))
    for aid, vec in schedule.items():
        assert sum(vec) <= delta, (
            f"agent {aid} accrued cumulative delay {sum(vec)} > budget delta={delta}"
        )


# --------------------------------------------------------------- test 2


def _two_agent_vertex_conflict_plan() -> Plan:
    """Two agents arriving at the shared vertex (1, 0) at the same wall-clock tick."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 0)),
        Agent(id=1, start=(1, 1), goal=(1, -1)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
        Path(agent_id=1, vertices=[(1, 1), (1, 0), (1, -1)]),
    ]
    return Plan.from_paths(agents, paths)


def test_adversarial_chooses_gap_reducing_agent() -> None:
    """The adversary must stall the earlier arriver to close a gap-1 pair."""
    # Skew the conflict so agent 0 arrives at (1, 0) at t=1 and agent 1 at t=2.
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 0)),
        Agent(id=1, start=(1, 2), goal=(1, -1)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
        Path(agent_id=1, vertices=[(1, 2), (1, 1), (1, 0), (1, -1)]),
    ]
    plan = Plan.from_paths(agents, paths)
    # Δ=1 makes the pair (agent 0 at t=1, agent 1 at t=2) unsafe (|t_i - t_j|=1 ≤ Δ).
    schedule = HeuristicAdversaryDelay(delta=1).sample(plan, np.random.default_rng(0))
    # Agent 0 is the earlier arriver — stalling it reduces the wall-clock gap.
    # Agent 1 is the later arriver — stalling it would *increase* the gap, so
    # the adversary must leave it alone.
    assert sum(schedule[0]) > 0, "earlier arriver (agent 0) should have been stalled"
    assert sum(schedule[1]) == 0, "later arriver (agent 1) should not have been stalled"


# --------------------------------------------------------------- test 3


def test_adversarial_no_stall_when_no_unsafe_pair() -> None:
    """With no unsafe pair to attack, the adversary should never stall."""
    # Two agents on parallel disjoint corridors — no shared vertex at any time.
    agents = [
        Agent(id=0, start=(0, 0), goal=(3, 0)),
        Agent(id=1, start=(0, 5), goal=(3, 5)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0)]),
        Path(agent_id=1, vertices=[(0, 5), (1, 5), (2, 5), (3, 5)]),
    ]
    plan = Plan.from_paths(agents, paths)
    schedule = HeuristicAdversaryDelay(delta=2).sample(plan, np.random.default_rng(0))
    assert all(sum(v) == 0 for v in schedule.values()), (
        f"adversary stalled despite no unsafe pair: {schedule}"
    )


# --------------------------------------------------------------- misc


def test_adversarial_rejects_negative_delta() -> None:
    with pytest.raises(ValueError):
        HeuristicAdversaryDelay(delta=-1)


def test_adversarial_with_delta_zero_emits_all_zeros() -> None:
    plan = _two_agent_vertex_conflict_plan()
    schedule = HeuristicAdversaryDelay(delta=0).sample(plan, np.random.default_rng(0))
    assert all(sum(v) == 0 for v in schedule.values())
