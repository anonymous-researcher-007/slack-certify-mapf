"""Hypothesis-based invariants for the per-conflict risk formula.

These three properties together pin the corrected probabilistic
collision formula from multiple angles. Any future regression that
re-introduces the original sign error would fail at least two of
them:

* :func:`test_risk_is_monotone_decreasing_in_gap` — risk is
  monotone-decreasing in the nominal gap, the property the user
  formula initially violated.
* :func:`test_risk_at_p_d_zero_is_indicator` — at ``p_d == 0`` the
  formula reduces to a deterministic indicator on the gap, which
  the buggy ``Pr[|D_i - D_j| < gap]`` formula evaluated to ``1``
  (always) when ``gap > 0``.
* :func:`test_risk_increases_with_p_d_at_low_range` — in the
  near-zero ``p_d`` regime where the linear-order term dominates,
  raising ``p_d`` does not decrease risk; the buggy formula was
  monotone-*decreasing* in ``p_d``.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from slackcertify.certify.probabilistic import per_conflict_risk
from slackcertify.core.conflict import VertexConflict


@pytest.mark.property
@given(
    p_d=st.floats(min_value=0.001, max_value=0.2),
    t_smaller=st.integers(min_value=0, max_value=15),
    base_gap=st.integers(min_value=0, max_value=20),
    delta_window=st.integers(min_value=0, max_value=3),
)
def test_risk_is_monotone_decreasing_in_gap(
    p_d: float, t_smaller: int, base_gap: int, delta_window: int
) -> None:
    """Widening the nominal gap cannot increase the realised-collision risk."""
    c_narrow = VertexConflict(
        agent_i=0, agent_j=1, vertex=(0, 0), t_i=t_smaller, t_j=t_smaller + base_gap
    )
    c_wider = VertexConflict(
        agent_i=0,
        agent_j=1,
        vertex=(0, 0),
        t_i=t_smaller,
        t_j=t_smaller + base_gap + 5,
    )
    risk_narrow = per_conflict_risk(c_narrow, p_d, delta_window=delta_window)
    risk_wider = per_conflict_risk(c_wider, p_d, delta_window=delta_window)
    # Tiny tolerance for floating-point noise; the inequality is exact
    # in real arithmetic.
    assert risk_wider <= risk_narrow + 1e-12, (
        f"monotone-decreasing violated at p_d={p_d}, t_smaller={t_smaller}, "
        f"base_gap={base_gap}, delta_window={delta_window}: "
        f"risk_narrow={risk_narrow}, risk_wider={risk_wider}"
    )


@pytest.mark.property
@given(
    t_smaller=st.integers(min_value=0, max_value=12),
    signed_gap=st.integers(min_value=0, max_value=12),
    delta_window=st.integers(min_value=0, max_value=5),
)
def test_risk_at_p_d_zero_is_indicator(t_smaller: int, signed_gap: int, delta_window: int) -> None:
    """At ``p_d == 0`` both delays are deterministically zero, so the
    unsafe event ``|(k_l - k_s) + signed_gap| <= delta_window`` reduces
    to the deterministic test ``signed_gap <= delta_window``."""
    t_larger = t_smaller + signed_gap
    c = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=t_smaller, t_j=t_larger)
    risk = per_conflict_risk(c, p_d=0.0, delta_window=delta_window)
    expected = 1.0 if signed_gap <= delta_window else 0.0
    assert risk == pytest.approx(expected, abs=1e-12), (
        f"indicator violated at t_smaller={t_smaller}, signed_gap={signed_gap}, "
        f"delta_window={delta_window}: risk={risk}, expected={expected}"
    )


@pytest.mark.property
@given(
    t_smaller=st.integers(min_value=1, max_value=10),
    signed_gap=st.integers(min_value=1, max_value=5),
    # `allow_subnormal=False` keeps scipy's binomial PMF from
    # OverflowError-ing on denormal probabilities near 1e-308.
    p_d_a=st.floats(
        min_value=0.0,
        max_value=0.01,
        allow_subnormal=False,
        allow_nan=False,
        allow_infinity=False,
    ),
    p_d_b=st.floats(
        min_value=0.0,
        max_value=0.01,
        allow_subnormal=False,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_risk_increases_with_p_d_at_low_range(
    t_smaller: int, signed_gap: int, p_d_a: float, p_d_b: float
) -> None:
    """At fixed configuration and within the near-zero monotone regime,
    raising ``p_d`` does not decrease the per-conflict risk.

    Restricted to ``p_d in [0, 0.01]`` so we stay below the
    critical point of the dominant linear-order term
    ``t_smaller · p · (1 - p)^(N - 1)``, which sits at
    ``1 / (t_smaller + t_larger)`` and is therefore at least 0.045
    for any ``(t_smaller, t_larger)`` we draw here.
    """
    assume(p_d_a < p_d_b)
    t_larger = t_smaller + signed_gap
    c = VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=t_smaller, t_j=t_larger)
    risk_a = per_conflict_risk(c, p_d=p_d_a)
    risk_b = per_conflict_risk(c, p_d=p_d_b)
    assert risk_b >= risk_a - 1e-12, (
        f"non-decreasing-in-p_d violated at t_smaller={t_smaller}, "
        f"signed_gap={signed_gap}, p_d_a={p_d_a}, p_d_b={p_d_b}: "
        f"risk_a={risk_a}, risk_b={risk_b}"
    )
