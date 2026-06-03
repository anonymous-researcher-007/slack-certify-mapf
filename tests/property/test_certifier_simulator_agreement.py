"""Empirical agreement between the probabilistic certifier and the simulator.

This test pins the semantic contract diagnosed in the Phase 7 audit
(``/tmp/delta_window_audit``): with ``delta=1``, the probabilistic
certifier's per-conflict predicate
``|(D_j(t_j) - D_i(t_i)) + signed_gap| <= 1`` covers every
integer-tick overlap of agents' occupation windows, which is exactly
what :mod:`slackcertify.simulate.executor` detects. Empirically the
realised-collision rate from ``monte_carlo_rollout`` should therefore
stay within the certified ``epsilon`` plus a 3-sigma Wilson upper
band.

Before the ``delta=1`` fix in the Phase 7 runners, the same plans
under ``delta_window=0`` would underestimate realised risk: a
nominal-gap-of-1 pair with one agent waiting one tick at the shared
vertex produces an executor-recorded vertex collision that the
``delta_window=0`` predicate misses, and the empirical rate could
exceed ``epsilon`` by roughly the per-step delay probability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.repair import slack_certify
from slackcertify.simulate.rollout import monte_carlo_rollout
from tests.property.strategies import cross_plans

_M_ROLLOUTS = 1000
_EPSILON = 0.10


@pytest.mark.property
@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(
    plan=cross_plans(),
    p_d=st.sampled_from([0.01, 0.1]),
)
def test_certifier_simulator_agreement(plan, p_d) -> None:
    """Empirically validate Proposition 1 against the executor simulator.

    For each Hypothesis-drawn plan:
      1. Certify with ``slack_certify(..., delta=1)`` in probabilistic mode.
      2. Run ``_M_ROLLOUTS`` Monte Carlo rollouts under
         ``BernoulliDelayModel(p_d)`` via :func:`monte_carlo_rollout`,
         which drives the real :class:`Executor`.
      3. Assert the empirical collision rate stays within ``epsilon``
         plus a 3-sigma Wilson upper band.
    """
    pi_prime, cert = slack_certify(plan, mode="probabilistic", p_d=p_d, epsilon=_EPSILON, delta=1)
    assert cert.verify(pi_prime)

    rng = np.random.default_rng(seed=20260515)
    model = BernoulliDelayModel(p_d=p_d)
    result = monte_carlo_rollout(pi_prime, model, K=_M_ROLLOUTS, rng=rng)
    empirical = 1.0 - result.success_rate

    sigma = math.sqrt(max(_EPSILON * (1.0 - _EPSILON), 1e-9) / _M_ROLLOUTS)
    bound = _EPSILON + 3.0 * sigma
    assert empirical <= bound, (
        f"Certifier/simulator disagreement: empirical={empirical:.4f} > "
        f"bound={bound:.4f} (epsilon={_EPSILON}, p_d={p_d}, "
        f"plan={pi_prime})"
    )
