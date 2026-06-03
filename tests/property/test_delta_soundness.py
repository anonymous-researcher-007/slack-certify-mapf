"""Empirical proof of Theorem 1 (Δ-soundness of the bounded certifier).

For every (plan, Δ) pair drawn from the strategy, this test:

1. runs :func:`slack_certify` in bounded mode at the chosen Δ to obtain
   a certified plan ``π'``;
2. samples a battery of bounded delay schedules whose per-agent
   cumulative delay never exceeds Δ — a mix of random draws and the
   deterministic worst-case adversary;
3. asserts the joint execution of ``π'`` under each schedule is
   collision-free.

If Theorem 1 holds, no schedule can elicit a collision.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.repair import slack_certify
from tests.property.strategies import cross_plans, simulate_with_delays

_SCHEDULES_PER_INSTANCE = 50  # enough to stress-test under the CI profile


@pytest.mark.property
@given(plan=cross_plans(), delta=st.integers(min_value=0, max_value=3))
def test_certified_plan_survives_bounded_delays(plan, delta) -> None:
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=delta)
    assert cert.verify(pi_prime)

    model = BoundedDelayModel(delta=delta)
    rng = np.random.default_rng(seed=20260514)
    for _ in range(_SCHEDULES_PER_INSTANCE):
        delays = model.sample(pi_prime, rng)
        assert simulate_with_delays(pi_prime, delays), (
            f"Theorem 1 violated: collision under bounded delay schedule {delays} "
            f"for delta={delta}, plan={pi_prime}"
        )


@pytest.mark.property
@given(plan=cross_plans(), delta=st.integers(min_value=0, max_value=3))
def test_certified_plan_survives_worst_case_adversary(plan, delta) -> None:
    from slackcertify.core.conflict import detect_conflicts

    pi_prime, _ = slack_certify(plan, mode="bounded", delta=delta)
    adversary_schedule = BoundedDelayModel(delta=delta).worst_case_against(
        pi_prime, detect_conflicts(pi_prime, delta=delta)
    )
    assert simulate_with_delays(pi_prime, adversary_schedule), (
        f"Theorem 1 violated under the deterministic worst-case adversary "
        f"with delta={delta} (schedule={adversary_schedule})"
    )


@pytest.mark.property
@given(plan=cross_plans())
def test_zero_delta_is_no_op_for_already_safe_plans(plan) -> None:
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=0)
    # Cross plans are nominally conflict-free at delta=0 by construction,
    # so no waits should ever be inserted.
    assert cert.total_wait_inserted == 0
    assert cert.verify(pi_prime)
