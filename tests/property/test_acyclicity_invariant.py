"""Empirical proof of Lemma 1 (the certified TPG remains acyclic).

For every (plan, Δ) pair drawn from the strategy, this test runs
:func:`slack_certify` in bounded mode and asserts that the temporal-plan
graph of the certified plan is still a DAG. Lemma 1 of the paper claims
exactly this; the property test exercises it on every instance the
strategy generates.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slackcertify.core.tpg import build_tpg
from slackcertify.repair import slack_certify
from tests.property.strategies import cross_plans


@pytest.mark.property
@given(plan=cross_plans(), delta=st.integers(min_value=0, max_value=3))
def test_certified_tpg_is_acyclic(plan, delta) -> None:
    pi_prime, _ = slack_certify(plan, mode="bounded", delta=delta)
    tpg = build_tpg(pi_prime)
    assert tpg.is_acyclic(), (
        f"Lemma 1 violated: certified TPG is cyclic at delta={delta} " f"for plan={pi_prime}"
    )


@pytest.mark.property
@given(
    plan=cross_plans(),
    p_d=st.floats(min_value=0.001, max_value=0.05),
)
def test_certified_tpg_acyclic_in_probabilistic_mode(plan, p_d) -> None:
    # epsilon=0.10 (permissive smoke); reachable under the corrected
    # monotone-decreasing formula. The certifier now actually shifts
    # on shared-cell instances, and the acyclicity invariant must
    # still hold across every example.
    pi_prime, _ = slack_certify(plan, mode="probabilistic", p_d=p_d, epsilon=0.10)
    assert build_tpg(pi_prime).is_acyclic()


@pytest.mark.property
@given(plan=cross_plans(), delta=st.integers(min_value=0, max_value=3))
def test_per_shift_tpg_acyclicity(plan, delta) -> None:
    """Strengthens Lemma 1: TPG is acyclic after **every** intra-round shift.

    The paper's Lemma 1 claims per-round acyclicity (i.e., at the
    boundaries of the outer loop). The implementation actually
    maintains a strictly stronger invariant: the TPG is acyclic
    after every single ``_apply_repair`` call, not just at the end
    of a round. This test instruments
    :func:`slackcertify.repair.algorithm._apply_repair` via
    ``unittest.mock.patch`` to capture the post-shift plan after
    every call and asserts acyclicity on each captured state.

    Pinned by
    :func:`tests.unit.test_repair_batch.test_intra_round_acyclicity`
    on a single 4-agent hub fixture; this property variant exercises
    the invariant across every plan the Hypothesis strategy draws.
    """
    from unittest.mock import patch

    import slackcertify.repair.algorithm as algo  # noqa: PLC0415 - test-only

    real_apply = algo._apply_repair
    captured_post_shift: list = []

    def _recording(*args: object, **kwargs: object):
        """Capture (post-shift plan, conflict) tuples for off-line inspection."""
        new_plan = real_apply(*args, **kwargs)
        # Positional args: (plan, c, w, mode, delta, p_d, epsilon, budget_alloc).
        conflict = args[1] if len(args) > 1 else kwargs.get("c")
        captured_post_shift.append((new_plan, conflict))
        return new_plan

    with patch.object(algo, "_apply_repair", side_effect=_recording):
        slack_certify(plan, mode="bounded", delta=delta)

    # Every captured intermediate TPG must be acyclic — strictly
    # stronger than the per-round check the paper's Lemma 1 proves.
    for i, (intermediate_plan, conflict) in enumerate(captured_post_shift):
        assert build_tpg(intermediate_plan).is_acyclic(), (
            f"per-shift acyclicity violated after intra-round shift "
            f"#{i + 1} on conflict {conflict} (delta={delta})"
        )
