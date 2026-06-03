"""Unit tests for slackcertify.certify.probabilistic.

The 2-agent fixture is hand-built so that exactly one shared-vertex pair
(visit to (2, 0) at agent-0 t=2 and agent-1 t=3) contributes to the
union bound. The per-conflict closed form therefore equals the plan-level
union bound, which lets us compare it to a Monte-Carlo estimate of
``Pr[|D_0(2) - D_1(3)| < 1]``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slackcertify.certify.probabilistic import (
    is_epsilon_certified,
    per_conflict_risk,
    plan_risk_upper_bound,
    unsafe_pairs_probabilistic,
)
from slackcertify.core.conflict import VertexConflict, detect_conflicts
from slackcertify.core.plan import Agent, Path, Plan


def _two_agent_single_pair_plan() -> Plan:
    """One shared-vertex pair at (2, 0): agent-0 t=2, agent-1 t=3, gap=1."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(3, 0)),
        Agent(id=1, start=(2, 3), goal=(2, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0)]),
        Path(agent_id=1, vertices=[(2, 3), (2, 2), (2, 1), (2, 0)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_per_conflict_risk_invalid_p_d() -> None:
    c = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=1, t_j=2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        per_conflict_risk(c, p_d=-0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        per_conflict_risk(c, p_d=1.5)


@pytest.mark.unit
def test_per_conflict_risk_is_zero_when_p_d_zero_and_gap_positive() -> None:
    # With p_d = 0 both delays are pinned at 0, so the realised arrival
    # gap equals the nominal gap (1) which is > delta_window = 0; no
    # point-collision is possible.
    c = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3)
    assert per_conflict_risk(c, p_d=0.0) == 0.0


@pytest.mark.unit
def test_per_conflict_risk_p_d_one_with_unit_gap_is_zero() -> None:
    # With p_d = 1 every step delays, so D_0(2) = 2, D_1(3) = 3,
    # (d_j - d_i) + signed_gap = (3 - 2) + 1 = 2 != 0 -> no point collision.
    c = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3)
    assert per_conflict_risk(c, p_d=1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_per_conflict_risk_at_gap_zero_is_binomial_coincidence() -> None:
    # signed_gap = 0 means the unsafe event reduces to d_j == d_i, i.e.
    # the two Binomial(2, 0.5) draws coincide: Σ_k P(B=k)^2 ≈ 0.375.
    from scipy.stats import binom as _binom

    c = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=2)
    expected = sum(float(_binom(2, 0.5).pmf(k)) ** 2 for k in range(3))
    assert per_conflict_risk(c, p_d=0.5) == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_plan_risk_upper_bound_equals_per_conflict_for_single_pair() -> None:
    plan = _two_agent_single_pair_plan()
    pairs = detect_conflicts(plan, delta=plan.makespan)
    assert len(pairs) == 1
    p_d = 0.3
    assert plan_risk_upper_bound(plan, p_d) == pytest.approx(
        per_conflict_risk(pairs[0], p_d), rel=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize("p_d", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("delta_window", [0, 1, 2])
def test_union_bound_matches_monte_carlo_within_3_sigma(p_d: float, delta_window: int) -> None:
    # The plan has a unique conflict pair with t_i = 2, t_j = 3
    # (signed_gap = 1), so the plan-level upper bound coincides with the
    # per-conflict probability.
    #
    # Collision (Δ-window) condition derivation. Wall-clock arrivals are
    #   t_i + d_i  and  t_j + d_j
    # and the unsafe event is |arrival_i - arrival_j| ≤ Δ_w. Substituting
    # signed_gap = t_j - t_i ≥ 0:
    #   |(t_i + d_i) - (t_j + d_j)|
    #     = |(d_i - d_j) - signed_gap|
    #     = |(d_j - d_i) + signed_gap|.
    # The MC condition below uses the latter form so it matches the
    # closed-form `_binom_diff_at_offset` line-for-line.
    plan = _two_agent_single_pair_plan()
    closed_form = plan_risk_upper_bound(plan, p_d, delta_window=delta_window)

    rng = np.random.default_rng(20260514)
    n_samples = 10_000
    signed_gap = 1  # t_j - t_i for the fixture
    d_i = rng.binomial(n=2, p=p_d, size=n_samples)
    d_j = rng.binomial(n=3, p=p_d, size=n_samples)
    empirical = float(np.mean(np.abs((d_j - d_i) + signed_gap) <= delta_window))

    # 3-sigma binomial confidence band; small 1/n floor guards the
    # degenerate cases where the formula sits exactly at 0 or 1.
    se = math.sqrt(max(closed_form * (1.0 - closed_form), 1e-9) / n_samples)
    tolerance = max(3.0 * se, 1e-3)
    assert abs(empirical - closed_form) < tolerance, (
        f"p_d={p_d}, delta_window={delta_window}: "
        f"empirical={empirical}, closed-form={closed_form}, "
        f"|diff|={abs(empirical - closed_form)} >= 3-sigma={tolerance}"
    )


@pytest.mark.unit
def test_is_epsilon_certified_threshold() -> None:
    plan = _two_agent_single_pair_plan()
    p_d = 0.3
    bound = plan_risk_upper_bound(plan, p_d)
    assert is_epsilon_certified(plan, p_d, epsilon=bound + 1e-9)
    assert not is_epsilon_certified(plan, p_d, epsilon=max(0.0, bound - 1e-3))


@pytest.mark.unit
def test_unsafe_pairs_filters_by_budget() -> None:
    plan = _two_agent_single_pair_plan()
    p_d = 0.3
    risk = plan_risk_upper_bound(plan, p_d)
    # Budget above the per-conflict risk -> nothing flagged.
    assert unsafe_pairs_probabilistic(plan, p_d, per_conflict_budget=risk + 1e-9) == []
    # Budget strictly below -> the single pair is flagged.
    flagged = unsafe_pairs_probabilistic(plan, p_d, per_conflict_budget=max(0.0, risk - 1e-3))
    assert len(flagged) == 1


@pytest.mark.unit
def test_invalid_inputs_rejected() -> None:
    plan = _two_agent_single_pair_plan()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plan_risk_upper_bound(plan, p_d=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        is_epsilon_certified(plan, p_d=0.1, epsilon=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        unsafe_pairs_probabilistic(plan, p_d=0.1, per_conflict_budget=-0.1)
