"""Unit tests for slackcertify.delay.bernoulli."""

from __future__ import annotations

import numpy as np
import pytest

from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.bernoulli import BernoulliDelayModel


def _long_single_agent_plan(steps: int) -> Plan:
    verts = [(i, 0) for i in range(steps + 1)]
    return Plan.from_paths(
        agents=[Agent(id=0, start=verts[0], goal=verts[-1])],
        paths=[Path(agent_id=0, vertices=verts)],
    )


@pytest.mark.unit
@pytest.mark.parametrize("p_d", [-0.1, 1.0, 1.5])
def test_invalid_p_d_rejected(p_d: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        BernoulliDelayModel(p_d=p_d)


@pytest.mark.unit
def test_sample_lengths_and_values() -> None:
    plan = _long_single_agent_plan(steps=8)
    rng = np.random.default_rng(0)
    sched = BernoulliDelayModel(p_d=0.5).sample(plan, rng)
    assert len(sched[0]) == 8
    assert all(d in (0, 1) for d in sched[0])


@pytest.mark.unit
def test_sample_empirical_mean_close_to_p_d() -> None:
    plan = _long_single_agent_plan(steps=2000)
    rng = np.random.default_rng(1234)
    sched = BernoulliDelayModel(p_d=0.3).sample(plan, rng)
    mean = float(np.mean(sched[0]))
    assert abs(mean - 0.3) < 0.05, f"empirical mean={mean}"


@pytest.mark.unit
def test_sample_reproducibility() -> None:
    plan = _long_single_agent_plan(steps=20)
    model = BernoulliDelayModel(p_d=0.4)
    a = model.sample(plan, np.random.default_rng(99))
    b = model.sample(plan, np.random.default_rng(99))
    assert a == b


@pytest.mark.unit
def test_sample_zero_makespan_returns_empty() -> None:
    plan = Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(0, 0))],
        paths=[Path(agent_id=0, vertices=[(0, 0)])],
    )
    sched = BernoulliDelayModel(p_d=0.5).sample(plan, np.random.default_rng(0))
    assert sched == {0: []}


@pytest.mark.unit
def test_cumulative_distribution_is_binomial() -> None:
    dist = BernoulliDelayModel(p_d=0.25).cumulative_delay_distribution(20)
    assert abs(float(dist.mean()) - 5.0) < 1e-9
    assert abs(float(dist.var()) - 20 * 0.25 * 0.75) < 1e-9
    # Frozen distribution exposes pmf(...).
    assert abs(float(dist.pmf(5)) - float(dist.pmf(5))) == 0


@pytest.mark.unit
def test_cumulative_distribution_rejects_negative_t() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BernoulliDelayModel(p_d=0.1).cumulative_delay_distribution(-1)
