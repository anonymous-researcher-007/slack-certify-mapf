"""Unit tests for slackcertify.simulate.rollout and metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.simulate.metrics import wilson_ci
from slackcertify.simulate.rollout import monte_carlo_rollout


def _single_agent_plan(steps: int = 4) -> Plan:
    verts = [(i, 0) for i in range(steps + 1)]
    return Plan.from_paths(
        agents=[Agent(id=0, start=verts[0], goal=verts[-1])],
        paths=[Path(agent_id=0, vertices=verts)],
    )


def _two_disjoint_agents_plan() -> Plan:
    return Plan.from_paths(
        agents=[
            Agent(id=0, start=(0, 0), goal=(2, 0)),
            Agent(id=1, start=(0, 1), goal=(2, 1)),
        ],
        paths=[
            Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
            Path(agent_id=1, vertices=[(0, 1), (1, 1), (2, 1)]),
        ],
    )


@pytest.mark.unit
def test_rollout_is_deterministic_for_same_seed() -> None:
    plan = _single_agent_plan(6)
    model = BernoulliDelayModel(p_d=0.3)
    a = monte_carlo_rollout(plan, model, K=20, rng=np.random.default_rng(123))
    b = monte_carlo_rollout(plan, model, K=20, rng=np.random.default_rng(123))
    assert a.success_rate == b.success_rate
    assert a.mean_executed_soc == b.mean_executed_soc
    assert a.n_collisions_per_rollout == b.n_collisions_per_rollout


@pytest.mark.unit
def test_rollout_disjoint_agents_always_succeed() -> None:
    plan = _two_disjoint_agents_plan()
    rng = np.random.default_rng(0)
    result = monte_carlo_rollout(plan, BernoulliDelayModel(p_d=0.5), K=50, rng=rng)
    assert result.success_rate == 1.0
    assert result.n_successful == 50
    assert all(c == 0 for c in result.n_collisions_per_rollout)


@pytest.mark.unit
def test_rollout_aggregates_match_inputs() -> None:
    plan = _single_agent_plan(3)
    rng = np.random.default_rng(0)
    result = monte_carlo_rollout(plan, BoundedDelayModel(delta=2), K=10, rng=rng)
    assert result.n_rollouts == 10
    assert 0 <= result.n_successful <= 10
    assert len(result.n_collisions_per_rollout) == 10
    assert result.mean_executed_soc >= 3.0  # each rollout has SoC >= path makespan


@pytest.mark.unit
def test_rollout_rejects_zero_k() -> None:
    plan = _single_agent_plan()
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_rollout(plan, BernoulliDelayModel(p_d=0.1), K=0)


@pytest.mark.unit
def test_wilson_ci_contains_true_proportion() -> None:
    # 95% CI around 0.5 with n=200 should comfortably include 0.5.
    lo, hi = wilson_ci(100, 200, alpha=0.05)
    assert lo < 0.5 < hi
    # CI shrinks with n.
    lo10, hi10 = wilson_ci(5, 10)
    lo1000, hi1000 = wilson_ci(500, 1000)
    assert (hi10 - lo10) > (hi1000 - lo1000)


@pytest.mark.unit
def test_wilson_ci_endpoints() -> None:
    lo0, hi0 = wilson_ci(0, 100)
    lo1, hi1 = wilson_ci(100, 100)
    assert lo0 == pytest.approx(0.0)
    assert hi1 == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("k", "n", "alpha"),
    [(0, 0, 0.05), (-1, 10, 0.05), (5, 4, 0.05), (1, 10, -0.1)],
)
def test_wilson_ci_rejects_invalid_inputs(k: int, n: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        wilson_ci(k, n, alpha)


@pytest.mark.unit
def test_rollout_collision_rate_bounded_above_by_one() -> None:
    plan = _single_agent_plan(2)
    rng = np.random.default_rng(0)
    result = monte_carlo_rollout(plan, BernoulliDelayModel(p_d=0.05), K=30, rng=rng)
    assert 0.0 <= result.success_rate <= 1.0
    lo, hi = result.wilson_ci_95
    eps = 1e-9
    assert 0.0 <= lo <= result.success_rate + eps
    assert result.success_rate - eps <= hi <= 1.0 + eps
    assert math.isfinite(result.mean_executed_makespan)
