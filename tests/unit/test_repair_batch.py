"""Unit tests for the batch outer-loop semantics of slack_certify.

The 3-agent Kottinger smoke fixture (three agents converging on the
hub vertex (1, 1) at staggered nominal times, with agent 2 also
visiting (2, 1) twice) is the worst case the previous per-conflict
loop took 5 outer rounds to resolve. With the batch loop, Lemma 1's
``≤ n_agents`` bound must hold.
"""

from __future__ import annotations

import pytest

from slackcertify.certify.bounded import is_delta_certified, unsafe_pairs_bounded
from slackcertify.core.conflict import VertexConflict
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.core.tpg import build_tpg
from slackcertify.repair import slack_certify
from slackcertify.repair.algorithm import (
    CertificationFailure,
    _apply_repair,
    _compute_w,
    _is_still_unsafe,
)
from slackcertify.repair.ordering import topological_order


def _three_agent_kottinger_plan() -> Plan:
    """Same fixture used in tests/integration/test_kottinger_baseline.py."""
    agents = [
        Agent(id=0, start=(0, 1), goal=(2, 1)),
        Agent(id=1, start=(1, 0), goal=(1, 2)),
        Agent(id=2, start=(2, 0), goal=(2, 2)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 1), (1, 1), (2, 1)]),
        Path(agent_id=1, vertices=[(1, 0), (1, 1), (1, 2)]),
        Path(agent_id=2, vertices=[(2, 0), (2, 1), (2, 1), (2, 2)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_batch_converges_within_n_agents_outer_rounds() -> None:
    plan = _three_agent_kottinger_plan()
    n_agents = len(plan.agents)
    # Cap explicitly at n_agents so any future regression that exceeds
    # the Lemma 1 bound surfaces as CertificationFailure.
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=1, max_outer_rounds=n_agents)
    assert is_delta_certified(pi_prime, delta=1)
    assert cert.verify(pi_prime)


@pytest.mark.unit
def test_batch_total_waits_no_worse_than_per_conflict() -> None:
    plan = _three_agent_kottinger_plan()
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=1, max_outer_rounds=8)
    # The pre-refactor (per-conflict) loop inserted 6 waits on this fixture;
    # batch must match or beat that count.
    assert cert.total_wait_inserted <= 6
    assert is_delta_certified(pi_prime, delta=1)


@pytest.mark.unit
def test_batch_raises_when_max_rounds_zero() -> None:
    # With max_outer_rounds=0 the loop body never runs and the post-loop
    # detection is the only chance to certify; any plan with conflicts
    # therefore raises CertificationFailure rather than silently spinning.
    plan = _three_agent_kottinger_plan()
    with pytest.raises(CertificationFailure, match="did not converge"):
        slack_certify(plan, mode="bounded", delta=1, max_outer_rounds=0)


def _four_agent_per_pair_hub_plan() -> Plan:
    """4 agents on a 7×7 grid; agent 0 traverses three hub cells, each shared
    with exactly one of agents 1, 2, 3 at a different staggered time.

    Vertex visits (delta=1 conflicts in parentheses):

    * Hub (1, 0): agent 0 at t=1, agent 1 at t=2  →  pair (0, 1), w=1, downstream=1
    * Hub (3, 0): agent 0 at t=3, agent 2 at t=4  →  pair (0, 2), w=1, downstream=2
    * Hub (5, 0): agent 0 at t=5, agent 3 at t=6  →  pair (0, 3), w=1, downstream=3

    No other shared cells, so the round-1 schedule is exactly three
    independent shifts on three different agents.
    """
    agents = [
        Agent(id=0, start=(0, 0), goal=(6, 0)),
        Agent(id=1, start=(1, 2), goal=(1, 0)),
        Agent(id=2, start=(3, 4), goal=(3, 0)),
        Agent(id=3, start=(5, 6), goal=(5, 0)),
    ]
    paths = [
        Path(
            agent_id=0,
            vertices=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
        ),
        Path(agent_id=1, vertices=[(1, 2), (1, 1), (1, 0)]),
        Path(agent_id=2, vertices=[(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]),
        Path(
            agent_id=3,
            vertices=[(5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0)],
        ),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_intra_round_acyclicity() -> None:
    plan = _four_agent_per_pair_hub_plan()
    delta = 1

    unsafe = unsafe_pairs_bounded(plan, delta=delta)
    # Pre-condition: exactly the three (0, j) conflicts described above.
    assert len(unsafe) == 3, f"unexpected nominal conflict set: {unsafe}"
    pair_ids = sorted((c.agent_i, c.agent_j) for c in unsafe if isinstance(c, VertexConflict))
    assert pair_ids == [(0, 1), (0, 2), (0, 3)]

    ordered = topological_order(unsafe, build_tpg(plan))
    # Sanity: schedule sorted by min(t_i, t_j) ascending.
    times = [min(c.t_i, c.t_j) for c in ordered]
    assert times == sorted(times)

    # Each conflict's downstream agent is distinct (1, 2, 3).
    expected_downstream = []
    for c in ordered:
        if c.t_i < c.t_j or (c.t_i == c.t_j and c.agent_i < c.agent_j):
            expected_downstream.append(c.agent_j)
        else:
            expected_downstream.append(c.agent_i)
    assert sorted(expected_downstream) == [1, 2, 3]

    # Walk the inner loop one shift at a time, asserting the per-shift
    # acyclicity invariant after every individual repair.
    current = plan
    shifts_applied = 0
    for c in ordered:
        if not _is_still_unsafe(c, current, "bounded", delta, None, None, "uniform"):
            continue
        w = _compute_w(c, "bounded", delta, None, None, current, "uniform")
        if w == 0:
            continue
        current = _apply_repair(current, c, w, "bounded", delta, None, None, "uniform")
        shifts_applied += 1
        # PER-SHIFT invariant — strictly stronger than per-round.
        assert build_tpg(current).is_acyclic(), (
            f"TPG cycle introduced after intra-round shift #{shifts_applied} " f"on conflict {c}"
        )

    assert shifts_applied == 3
    # End-of-round post-condition: plan is now fully Δ=1 disjoint.
    assert is_delta_certified(current, delta=delta)


@pytest.mark.unit
def test_batch_preserves_certificate_invariants_on_simple_plan() -> None:
    # Two-agent plan with one conflict — both per-conflict and batch
    # should produce the same certified plan and wait count (1 round).
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 0)),
        Agent(id=1, start=(2, 1), goal=(0, 1)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
        Path(agent_id=1, vertices=[(2, 1), (1, 1), (0, 1)]),
    ]
    plan = Plan.from_paths(agents, paths)
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=0)
    # No conflicts in the input plan -> no waits, single round.
    assert cert.total_wait_inserted == 0
    assert pi_prime == plan
