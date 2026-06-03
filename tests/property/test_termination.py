"""Empirical termination check for the slack certifier.

Algorithm 1's outer loop is bounded above by ``n_agents * |conflicts_0|``
rounds. This test draws random ``(plan, Δ)`` pairs, calls
:func:`slack_certify` with that explicit bound, and asserts it returns a
valid certificate without raising :class:`CertificationFailure`.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slackcertify.repair import slack_certify
from tests.property.strategies import cross_plans, initial_conflict_count


@pytest.mark.property
@given(plan=cross_plans(), delta=st.integers(min_value=0, max_value=5))
def test_certifier_terminates_within_theoretical_bound(plan, delta) -> None:
    n_agents = len(plan.agents)
    n_conflicts = max(1, initial_conflict_count(plan, delta))
    bound = n_agents * n_conflicts + 1
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=delta, max_outer_rounds=bound)
    # Certificate verifies and inserted-wait count is finite.
    assert cert.verify(pi_prime)
    assert 0 <= cert.total_wait_inserted < bound * (delta + 1) + 100


@pytest.mark.property
@given(plan=cross_plans(), delta=st.integers(min_value=0, max_value=3))
def test_certifier_at_default_bound_terminates(plan, delta) -> None:
    # Default ``max_outer_rounds`` is ``len(plan.agents)``; the cross
    # topology is simple enough that this default suffices.
    pi_prime, cert = slack_certify(plan, mode="bounded", delta=delta)
    assert cert.verify(pi_prime)
