"""Unit tests for slackcertify.repair.budget."""

from __future__ import annotations

import math

import pytest

from slackcertify.core.conflict import VertexConflict
from slackcertify.repair.budget import risk_proportional_budget, uniform_budget


@pytest.mark.unit
def test_uniform_budget_sums_to_epsilon() -> None:
    epsilon = 0.1
    n = 5
    share = uniform_budget(epsilon, n)
    assert share * n == pytest.approx(epsilon)


@pytest.mark.unit
def test_uniform_budget_zero_conflicts_returns_inf() -> None:
    assert uniform_budget(0.05, 0) == math.inf


@pytest.mark.unit
@pytest.mark.parametrize(
    ("eps", "n"),
    [(-0.01, 3), (0.05, -1)],
)
def test_uniform_budget_rejects_negatives(eps: float, n: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        uniform_budget(eps, n)


@pytest.mark.unit
def test_risk_proportional_sums_to_epsilon_when_risks_positive() -> None:
    cs = [
        VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3),
        VertexConflict(agent_i=0, agent_j=2, vertex=(1, 1), t_i=4, t_j=5),
    ]
    eps = 0.1
    budgets = risk_proportional_budget(eps, cs, p_d=0.3)
    assert sum(budgets.values()) == pytest.approx(eps, rel=1e-12)


@pytest.mark.unit
def test_risk_proportional_falls_back_to_uniform_when_all_zero() -> None:
    # Force structurally-zero raw risk: t_smaller = 0 makes
    # D_smaller(0) = 0 deterministic, so the unsafe event
    # |k_l - 0 + signed_gap| <= 0 reduces to k_l = -signed_gap which is
    # impossible for positive signed_gap. Under the corrected formula
    # this is the only way to get raw risk exactly zero (gap=0 same-
    # time visits have non-zero Binomial-coincidence risk).
    cs = [
        VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=0, t_j=3),
        VertexConflict(agent_i=0, agent_j=2, vertex=(1, 1), t_i=0, t_j=5),
    ]
    eps = 0.1
    budgets = risk_proportional_budget(eps, cs, p_d=0.5)
    assert all(b == pytest.approx(eps / len(cs)) for b in budgets.values())


@pytest.mark.unit
def test_risk_proportional_is_proportional_to_raw_risk() -> None:
    # Two conflicts with different gaps at p_d=0.5. The per-conflict
    # risk formula is monotone-decreasing in the nominal gap, so the
    # *smaller*-gap conflict gets the larger raw risk and therefore the
    # larger budget under the proportional allocator.
    from slackcertify.certify.probabilistic import per_conflict_risk

    a = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3)
    b = VertexConflict(agent_i=0, agent_j=2, vertex=(1, 1), t_i=2, t_j=4)
    eps = 0.1
    budgets = risk_proportional_budget(eps, [a, b], p_d=0.5)
    risk_a = per_conflict_risk(a, 0.5)
    risk_b = per_conflict_risk(b, 0.5)
    # Allocation matches the raw-risk ratio, exactly.
    assert budgets[a] / budgets[b] == pytest.approx(risk_a / risk_b, rel=1e-12)
    assert sum(budgets.values()) == pytest.approx(eps, rel=1e-12)


@pytest.mark.unit
def test_risk_proportional_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        risk_proportional_budget(0.1, [], p_d=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        risk_proportional_budget(-0.1, [], p_d=0.3)


@pytest.mark.unit
def test_risk_proportional_empty_list_returns_empty_dict() -> None:
    assert risk_proportional_budget(0.1, [], p_d=0.3) == {}
