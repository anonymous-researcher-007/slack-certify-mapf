"""Edge-case unit tests for the RQ5 Bernoulli↔Persistent calibration.

:func:`calibrate_bernoulli_to_persistent` is used by the RQ5 runner to
turn a persistent (MAPF-DP-style) delay model into the equivalent
per-step Bernoulli probability ``p_d`` for the probabilistic certifier.
These tests pin the limit behaviour:

* ``p_trigger == 0``           ⇒ calibrated ``p_d == 0`` exactly,
* ``p_trigger`` near 1 with    ⇒ calibrated ``p_d`` is at most 1,
  ``stall_len ≥ plan_length``
* fixed plan length            ⇒ calibrated ``p_d`` is monotone
                                  non-decreasing in ``p_trigger``.
"""

from __future__ import annotations

import numpy as np
import pytest

from slackcertify.delay.calibration import calibrate_bernoulli_to_persistent
from slackcertify.delay.persistent import PersistentDelayModel


@pytest.mark.unit
def test_zero_trigger_gives_exactly_zero_p_d() -> None:
    """With ``p_trigger=0`` the persistent process never fires, so the
    empirical per-step blocking rate is identically zero."""
    rng = np.random.default_rng(0)
    p_d = calibrate_bernoulli_to_persistent(
        PersistentDelayModel(p_trigger=0.0, min_len=2, max_len=4),
        plan_length=20,
        rng=rng,
    )
    assert p_d == 0.0


@pytest.mark.unit
def test_near_unit_trigger_with_saturating_stall_clamped_below_one() -> None:
    """With ``p_trigger`` near 1 and ``min_len = max_len = plan_length``,
    the first step of every rollout fires a stall that fills the whole
    plan — so the empirical rate is at most 1 and effectively saturates
    to ~1 (Note: the function does not strictly clip; the RQ5 runner
    clips the value to ``BernoulliDelayModel``'s open-interval domain
    ``[0, 1)`` before passing it on)."""
    plan_length = 8
    rng = np.random.default_rng(0)
    # PersistentDelayModel rejects p_trigger == 1.0 (open interval),
    # so we use 0.999 — high enough that almost every rollout triggers
    # at t=0 and stays stalled for the full plan_length.
    p_d = calibrate_bernoulli_to_persistent(
        PersistentDelayModel(p_trigger=0.999, min_len=plan_length, max_len=plan_length),
        plan_length=plan_length,
        rng=rng,
    )
    assert 0.0 <= p_d <= 1.0
    # With the near-saturating trigger every rollout produces almost
    # ``plan_length`` delay steps, so the calibrated rate is close to 1.
    assert p_d >= 0.99


@pytest.mark.unit
def test_calibrated_p_d_is_monotone_in_p_trigger() -> None:
    """On a fixed plan length, raising ``p_trigger`` cannot lower the
    empirical blocking rate."""
    plan_length = 60
    triggers = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    calibrated: list[float] = []
    for p in triggers:
        # Seed each call identically so the comparison reflects only the
        # change in ``p_trigger``, not internal RNG drift.
        rng = np.random.default_rng(20260514)
        calibrated.append(
            calibrate_bernoulli_to_persistent(
                PersistentDelayModel(p_trigger=p, min_len=3, max_len=5),
                plan_length=plan_length,
                rng=rng,
            )
        )

    # All entries in [0, 1].
    assert all(0.0 <= v <= 1.0 for v in calibrated)
    # First entry is exactly zero (matches the dedicated test above).
    assert calibrated[0] == 0.0
    # Non-decreasing in ``p_trigger``.
    for lower, upper in zip(calibrated, calibrated[1:], strict=False):
        assert upper >= lower - 1e-12, (
            f"calibrated p_d should not decrease as p_trigger rises; "
            f"got sequence {calibrated} for triggers {triggers}"
        )
    # The high-trigger end is strictly greater than the all-zero start.
    assert calibrated[-1] > calibrated[0]
