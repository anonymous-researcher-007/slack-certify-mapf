"""Unit tests for slackcertify.repair.ordering."""

from __future__ import annotations

import numpy as np
import pytest

from slackcertify.certify.probabilistic import per_conflict_risk
from slackcertify.core.conflict import VertexConflict, detect_conflicts
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.core.tpg import build_tpg
from slackcertify.repair.ordering import (
    random_order,
    risk_descending_order,
    topological_order,
)


def _two_agent_plan() -> Plan:
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 1)),
        Agent(id=1, start=(1, 3), goal=(1, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 1), (2, 1)]),
        Path(agent_id=1, vertices=[(1, 3), (1, 2), (1, 1), (1, 0)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_topological_order_sorts_by_min_time() -> None:
    plan = _two_agent_plan()
    conflicts = detect_conflicts(plan, delta=plan.makespan)
    ordered = topological_order(conflicts, build_tpg(plan))
    times = [min(c.t_i, c.t_j) for c in ordered]
    assert times == sorted(times)


@pytest.mark.unit
def test_topological_order_respects_tpg_rank_within_same_time() -> None:
    plan = _two_agent_plan()
    tpg = build_tpg(plan)
    conflicts = detect_conflicts(plan, delta=plan.makespan)
    ordered = topological_order(conflicts, tpg)
    rank_of = {n: i for i, n in enumerate(tpg.topological_order())}
    # Within identical (min_time) groups, the anchor-rank must be sorted.
    grouped: dict[int, list[int]] = {}
    for c in ordered:
        early_agent = c.agent_i if c.t_i <= c.t_j else c.agent_j
        early_t = min(c.t_i, c.t_j)
        early_cell = c.vertex if isinstance(c, VertexConflict) else c.u
        from slackcertify.core.tpg import TPGNode

        node = TPGNode(early_agent, early_cell, early_t)
        grouped.setdefault(early_t, []).append(rank_of.get(node, 10**9))
    for ranks in grouped.values():
        assert ranks == sorted(ranks)


@pytest.mark.unit
def test_topological_order_is_a_permutation() -> None:
    plan = _two_agent_plan()
    conflicts = detect_conflicts(plan, delta=plan.makespan)
    ordered = topological_order(conflicts, build_tpg(plan))
    assert sorted(ordered, key=id) == sorted(conflicts, key=id)


@pytest.mark.unit
def test_risk_descending_sorts_by_descending_risk() -> None:
    p_d = 0.3
    cs = [
        VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=2),  # zero
        VertexConflict(agent_i=0, agent_j=2, vertex=(0, 0), t_i=2, t_j=3),  # positive
        VertexConflict(agent_i=0, agent_j=3, vertex=(0, 0), t_i=4, t_j=5),  # positive
    ]
    ordered = risk_descending_order(cs, p_d=p_d)
    risks = [per_conflict_risk(c, p_d) for c in ordered]
    assert risks == sorted(risks, reverse=True)


@pytest.mark.unit
def test_risk_descending_rejects_invalid_p_d() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        risk_descending_order([], p_d=1.5)


@pytest.mark.unit
def test_random_order_is_permutation() -> None:
    cs = [VertexConflict(agent_i=0, agent_j=1, vertex=(i, 0), t_i=i, t_j=i + 1) for i in range(5)]
    rng = np.random.default_rng(7)
    permuted = random_order(cs, rng)
    assert len(permuted) == len(cs)
    assert {id(c) for c in permuted} == {id(c) for c in cs}


@pytest.mark.unit
def test_random_order_reproducible_for_same_seed() -> None:
    cs = [VertexConflict(agent_i=0, agent_j=1, vertex=(i, 0), t_i=i, t_j=i + 1) for i in range(8)]
    a = random_order(cs, np.random.default_rng(42))
    b = random_order(cs, np.random.default_rng(42))
    assert [c.t_i for c in a] == [c.t_i for c in b]
