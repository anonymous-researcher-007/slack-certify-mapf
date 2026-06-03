"""Empirical sanity check of Proposition 1 (ε-bound under Bernoulli delays).

For every certified plan, this test draws ``M`` Bernoulli rollouts of
the per-step delay model and compares the empirical realised-collision
rate against the certified ε plus the 3-σ Wilson upper bound. If
Proposition 1 holds, the empirical rate stays under the bound.

Under the corrected (monotone-decreasing-in-gap) formula, a tight ε
is reachable in finite shifts; we use ``ε = 0.10`` here (permissive
smoke) so the certifier actually inserts waits when needed and the
empirical rate is bounded meaningfully.

We also pass ``delta = 1`` to ``slack_certify`` so the certifier
formula bounds ``Pr[|arrival diff| ≤ 1]``, matching the simulator's
detection of any single-tick interval overlap. With ``delta = 0``
the formula would bound only *point* coincidence, which underestimates
the simulator's empirical rate by roughly the per-step delay
probability — exactly the bug surfaced in PART C's first pass
(empirical 0.135 vs point-collision bound 0.123 at p_d≈0.04).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.repair import slack_certify
from tests.property.strategies import cross_plans, simulate_with_delays

_M_ROLLOUTS = 1500
# epsilon=0.10 (permissive smoke); reachable under the corrected
# monotone-decreasing formula.
_SMOKE_EPSILON = 0.10


@pytest.mark.property
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(
    plan=cross_plans(),
    p_d=st.floats(min_value=0.001, max_value=0.05),
)
def test_empirical_rate_below_three_sigma_band(plan, p_d) -> None:
    pi_prime, cert = slack_certify(
        plan, mode="probabilistic", p_d=p_d, epsilon=_SMOKE_EPSILON, delta=1
    )
    assert cert.verify(pi_prime)

    rng = np.random.default_rng(seed=20260514)
    model = BernoulliDelayModel(p_d=p_d)
    n_collisions = 0
    for _ in range(_M_ROLLOUTS):
        delays = model.sample(pi_prime, rng)
        if not simulate_with_delays(pi_prime, delays):
            n_collisions += 1
    empirical = n_collisions / _M_ROLLOUTS

    sigma = math.sqrt(max(_SMOKE_EPSILON * (1.0 - _SMOKE_EPSILON), 1e-9) / _M_ROLLOUTS)
    bound = _SMOKE_EPSILON + 3.0 * sigma
    assert empirical <= bound, (
        f"Proposition 1 violated: empirical={empirical:.4f} > "
        f"bound={bound:.4f} (epsilon={_SMOKE_EPSILON}, p_d={p_d}, "
        f"plan={pi_prime})"
    )
