"""Unit tests for slackcertify.delay.calibration and PersistentDelayModel."""

from __future__ import annotations

import numpy as np
import pytest

from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.delay.calibration import calibrate_bernoulli_to_persistent
from slackcertify.delay.persistent import PersistentDelayModel


def _single_agent_plan(steps: int) -> Plan:
    verts = [(i, 0) for i in range(steps + 1)]
    return Plan.from_paths(
        agents=[Agent(id=0, start=verts[0], goal=verts[-1])],
        paths=[Path(agent_id=0, vertices=verts)],
    )


@pytest.mark.unit
def test_persistent_zero_trigger_yields_no_delay() -> None:
    plan = _single_agent_plan(50)
    rng = np.random.default_rng(0)
    sched = PersistentDelayModel(p_trigger=0.0, min_len=2, max_len=4).sample(plan, rng)
    assert sum(sched[0]) == 0


@pytest.mark.unit
def test_persistent_validates_lengths() -> None:
    with pytest.raises(ValueError, match="min_len"):
        PersistentDelayModel(p_trigger=0.1, min_len=0, max_len=5)
    with pytest.raises(ValueError, match="max_len"):
        PersistentDelayModel(p_trigger=0.1, min_len=4, max_len=2)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        PersistentDelayModel(p_trigger=1.0, min_len=1, max_len=2)


@pytest.mark.unit
def test_persistent_reproducibility() -> None:
    plan = _single_agent_plan(40)
    model = PersistentDelayModel(p_trigger=0.05, min_len=3, max_len=5)
    a = model.sample(plan, np.random.default_rng(7))
    b = model.sample(plan, np.random.default_rng(7))
    assert a == b


@pytest.mark.unit
def test_calibration_returns_unit_interval() -> None:
    rng = np.random.default_rng(42)
    p_d = calibrate_bernoulli_to_persistent(
        PersistentDelayModel(p_trigger=0.05, min_len=2, max_len=4),
        plan_length=50,
        rng=rng,
    )
    assert 0.0 <= p_d <= 1.0


@pytest.mark.unit
def test_calibration_zero_trigger_gives_zero() -> None:
    rng = np.random.default_rng(0)
    p_d = calibrate_bernoulli_to_persistent(
        PersistentDelayModel(p_trigger=0.0, min_len=1, max_len=1),
        plan_length=20,
        rng=rng,
    )
    assert p_d == 0.0


@pytest.mark.unit
def test_calibration_close_to_expected_rate() -> None:
    # With p_trigger=0.5 and max_len=min_len=1, every trigger produces a
    # single delay step, so the empirical rate should hover around 0.5.
    rng = np.random.default_rng(123)
    p_d = calibrate_bernoulli_to_persistent(
        PersistentDelayModel(p_trigger=0.5, min_len=1, max_len=1),
        plan_length=200,
        rng=rng,
    )
    assert abs(p_d - 0.5) < 0.05, f"calibrated p_d={p_d}"


@pytest.mark.unit
def test_calibration_rejects_zero_length() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positive"):
        calibrate_bernoulli_to_persistent(
            PersistentDelayModel(p_trigger=0.1, min_len=1, max_len=2),
            plan_length=0,
            rng=rng,
        )
