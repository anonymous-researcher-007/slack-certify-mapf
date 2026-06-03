"""Probabilistic certifier under i.i.d. Bernoulli per-step delays.

For two visits ``(agent_i, t_i)`` and ``(agent_j, t_j)`` to the same
cell with ``t_i <= t_j`` (canonicalised inside the helpers), the
**realised collision probability** under cumulative-delay model
``D_i(t) \\sim \\mathrm{Binomial}(t, p_d)`` is

.. math::

    R_{ij}(t_i, t_j; p_d, \\Delta_w)
    \\;=\\; \\Pr\\!\\left[\\,
        \\bigl|\\,(t_i + D_i(t_i)) - (t_j + D_j(t_j))\\,\\bigr| \\;\\le\\; \\Delta_w
    \\,\\right],

i.e. the wall-clock arrival times must coincide (``Δ_w = 0``, point
collision) or be within ``Δ_w`` ticks of each other (Δ-window) for
a collision to be *realised*.

Equivalently, letting ``signed_gap = t_j - t_i >= 0`` and writing
``k_s = D_i(t_i)``, ``k_l = D_j(t_j)``:

.. math::

    R_{ij} = \\Pr\\!\\left[\\,\\bigl|(k_l - k_s) + \\text{signed\\_gap}\\bigr|
                           \\;\\le\\; \\Delta_w\\,\\right].

**Monotonicity.** This probability is **monotone-decreasing in the
nominal gap** ``|t_i - t_j|``: a larger nominal separation puts the
delay-difference target farther away from zero, which the cumulative
Binomial distributions struggle to reach. This is exactly the
property the certifier relies on — inserting waits *grows* the
nominal gap and so *reduces* realised-collision risk, restoring
slack insertion as a productive repair operation.

(Historical note. An earlier formula in this module computed
``Pr[|D_i(t_i) - D_j(t_j)| < |t_i - t_j|]``, which is the probability
that the realised order is *preserved* — the *complement* of a
collision under single-tick semantics. It was monotone-increasing in
the gap and made the probabilistic certifier loop forever on
shared-cell instances. The diagnostic that surfaced the bug lives at
``/tmp/probabilistic_monotonicity.md``.)

The plan-level :func:`plan_risk_upper_bound` is the union bound over
every shared-vertex / reverse-edge pair (every pair returned by
:func:`detect_conflicts` for ``delta = plan.makespan``).
"""

from __future__ import annotations

from functools import cache

from scipy.stats import binom

from slackcertify.core.conflict import (
    Conflict,
    VertexConflict,
    detect_conflicts,
    detect_conflicts_fast,
)
from slackcertify.core.plan import Plan

__all__ = [
    "is_epsilon_certified",
    "per_conflict_risk",
    "plan_risk_upper_bound",
    "plan_risk_upper_bound_fast",
    "unsafe_pairs_probabilistic",
]

# Window-selection constants for plan_risk_upper_bound_fast.
#
# TAU: a dropped (gap > W) conflict object is only ever skipped when the
# *maximum possible* per_conflict_risk at that gap (over the plan's anchor
# range) is below TAU. With TAU = 1e-18, even an adversarial plan with ~1e7
# conflict objects (n=200-scale) drops at most ~1e-7 * 1e-18 = ~1e-11 of total
# mass, far below the 1e-9 equivalence tolerance. (per_conflict_risk decays
# super-exponentially in the gap, so shrinking TAU costs only a few extra
# units of W.)
#
# WINDOW_MARGIN: extra gap units added to the analytically-minimal W as a
# conservative cushion ("when in doubt, larger W").
_RISK_WINDOW_TAU: float = 1e-18
_RISK_WINDOW_MARGIN: int = 3


def per_conflict_risk(c: Conflict, p_d: float, delta_window: int = 0) -> float:
    """Return the per-conflict realised-collision probability.

    Computes ``Pr[|(k_l - k_s) + signed_gap| <= delta_window]``
    exactly, where ``k_s ~ Binomial(t_smaller, p_d)``,
    ``k_l ~ Binomial(t_larger, p_d)``, and
    ``signed_gap = t_larger - t_smaller``.

    ``delta_window`` is the Δ-margin within which arrival-time
    coincidence is considered unsafe. ``delta_window = 0`` gives the
    *point*-collision probability (the two wall-clock arrival times
    exactly coincide); ``delta_window = Δ`` gives the Δ-disjointness
    realisation probability, matching the bounded-mode definition in
    §III Definition 1 of the paper. The two modes therefore share a
    single, unified Δ-margin semantics.

    The result is cached on the canonical key
    ``(t_smaller, t_larger, p_d, signed_gap, delta_window)``.

    Numerical validity: this function assumes ``p_d`` is a normal
    float in ``(0, 1)``. For ``p_d <= machine epsilon``, the
    binomial PMF underflows; callers should clamp ``p_d`` to a
    positive minimum (e.g. ``1e-10``) if they need to handle
    near-zero rates.

    Examples
    --------
    >>> from slackcertify.core.conflict import VertexConflict
    >>> # With p_d == 0 and a positive gap, both delays are pinned at
    >>> # zero so the realised-arrival gap equals the nominal gap; no
    >>> # point collision is possible.
    >>> per_conflict_risk(
    ...     VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=2, t_j=3),
    ...     p_d=0.0,
    ... )
    0.0
    """
    if not (0.0 <= p_d <= 1.0):
        raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
    if delta_window < 0:
        raise ValueError(f"delta_window must be non-negative, got {delta_window}")
    t_i, t_j = c.t_i, c.t_j
    t_smaller, t_larger = min(t_i, t_j), max(t_i, t_j)
    return _binom_diff_at_offset(
        t_smaller=t_smaller,
        t_larger=t_larger,
        p_d=float(p_d),
        signed_gap=t_larger - t_smaller,
        delta_window=int(delta_window),
    )


def plan_risk_upper_bound(plan: Plan, p_d: float, delta_window: int = 0) -> float:
    """Return the union-bound risk of ``plan`` under Bernoulli(``p_d``) delays.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
    >>> float(plan_risk_upper_bound(Plan.from_paths(a, p), p_d=0.1))
    0.0
    """
    if not (0.0 <= p_d <= 1.0):
        raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
    horizon = max(plan.makespan, 0)
    # detect_conflicts_fast is the window-aware EXACT equivalent of
    # detect_conflicts (same conflict set, validated to 1e-12 risk identity);
    # it removes the O(n^2 * makespan) bottleneck at delta=makespan that blocked
    # the probabilistic sweep. The risk value is unchanged.
    pairs = detect_conflicts_fast(plan, delta=horizon)
    return sum(per_conflict_risk(c, p_d, delta_window=delta_window) for c in pairs)


@cache
def _risk_window(p_d: float, delta_window: int, makespan: int) -> int:
    """Smallest gap window ``W`` such that every dropped pair is provably ~0.

    The original :func:`plan_risk_upper_bound` sums
    :func:`per_conflict_risk` over **every** shared-vertex / reverse-edge pair
    (``detect_conflicts(delta=makespan)``). We instead enumerate only pairs
    with ``|t_i - t_j| <= W`` and prove the omitted pairs contribute
    negligibly.

    *Derivation.* For a fixed nominal gap ``g`` the realised-collision
    probability ``per_conflict_risk`` is **monotone-decreasing in ``g``**
    (module docstring / ``_binom_diff_at_offset``), but it is **not** monotone
    in the absolute anchor ``t_smaller`` (it rises to a peak then falls). So we
    take, for each candidate ``g``, the *maximum over the anchor range*
    ``peak(g) = max_{t in [0, makespan]} per_conflict_risk(t, t + g)``. Because
    each summand is decreasing in ``g``, the pointwise maximum ``peak(g)`` is
    decreasing in ``g`` too. Every real conflict object has both endpoints in
    ``[0, makespan]`` (``detect_conflicts`` scans that range), so a dropped
    pair with gap ``g`` satisfies ``risk <= peak(g)``.

    ``W`` is the smallest ``g`` with ``peak(g) < TAU`` (binary search on the
    decreasing ``peak``), plus a conservative margin, clamped to ``makespan``
    (at which point the window equals the full scan and the result is exactly
    the original). Every dropped pair (gap ``> W``) then has
    ``risk <= peak(W + 1) <= peak(W) < TAU``; with ``TAU = 1e-18`` the total
    dropped mass stays far below the ``1e-9`` equivalence tolerance even for
    millions of pairs.

    Computed once per ``(p_d, delta_window, makespan)`` and cached. ``W`` grows
    with ``p_d`` (slower decay) exactly as required.
    """
    tau = _RISK_WINDOW_TAU
    if makespan <= 0:
        return 0

    def peak(g: int) -> float:
        """max_{t in [0, makespan]} per_conflict_risk(t, t + g)."""
        best = 0.0
        prev = -1.0
        for t in range(0, makespan + 1):
            r = per_conflict_risk(
                VertexConflict(agent_i=0, agent_j=1, vertex=(0, 0), t_i=t, t_j=t + g),
                p_d,
                delta_window=delta_window,
            )
            if r > best:
                best = r
            # risk(t, g) is 0 up to the first feasible anchor, rises to a single
            # peak, then falls; once it is both decreasing and below tau we are
            # past the peak and the remaining tail is even smaller.
            if r < prev and r < tau:
                break
            prev = r
        return best

    # peak(g) is decreasing in g; bracket the crossing by doubling.
    hi = 1
    while hi < makespan and peak(hi) >= tau:
        hi *= 2
    hi = min(hi, makespan)
    if peak(hi) >= tau:
        # Even gap == makespan does not fall below tau (very slow decay and/or
        # tiny makespan): fall back to the full scan — exact, just not faster.
        return makespan

    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if peak(mid) < tau:
            hi = mid
        else:
            lo = mid + 1
    return min(makespan, lo + _RISK_WINDOW_MARGIN)


def plan_risk_upper_bound_fast(plan: Plan, p_d: float, delta_window: int = 0) -> float:
    """Exact fast equivalent of :func:`plan_risk_upper_bound`.

    Returns the **same** union-bound risk (to within ``< 1e-9``) but enumerates
    only conflict pairs within a provably-sufficient gap window ``W`` chosen by
    :func:`_risk_window`, instead of the full ``O(makespan * n^2)`` scan over
    every shared-cell / reverse-edge coincidence. Pairs beyond ``W`` are
    omitted only when their realised-collision probability is below
    ``1e-18`` — see :func:`_risk_window` for the derivation and soundness
    argument.

    This is a drop-in accelerator for the soundness gate (Lemma 3); it does
    **not** replace :func:`plan_risk_upper_bound`, which remains the ground
    truth.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
    >>> float(plan_risk_upper_bound_fast(Plan.from_paths(a, p), p_d=0.1))
    0.0
    """
    if not (0.0 <= p_d <= 1.0):
        raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
    horizon = max(plan.makespan, 0)
    window = _risk_window(p_d, int(delta_window), horizon)
    pairs = detect_conflicts(plan, delta=window)
    return sum(per_conflict_risk(c, p_d, delta_window=delta_window) for c in pairs)


def is_epsilon_certified(plan: Plan, p_d: float, epsilon: float, delta_window: int = 0) -> bool:
    """Return ``True`` iff the plan's union-bound risk is at most ``epsilon``.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
    >>> is_epsilon_certified(Plan.from_paths(a, p), p_d=0.1, epsilon=0.05)
    True
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    return plan_risk_upper_bound(plan, p_d, delta_window=delta_window) <= epsilon


def unsafe_pairs_probabilistic(
    plan: Plan,
    p_d: float,
    per_conflict_budget: float,
    delta_window: int = 0,
) -> list[Conflict]:
    """Return every conflict whose per-conflict risk exceeds ``per_conflict_budget``."""
    if not (0.0 <= p_d <= 1.0):
        raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
    if per_conflict_budget < 0.0:
        raise ValueError(f"per_conflict_budget must be non-negative, got {per_conflict_budget}")
    horizon = max(plan.makespan, 0)
    # detect_conflicts_fast: exact-equivalent window-aware scan (same set).
    return [
        c
        for c in detect_conflicts_fast(plan, delta=horizon)
        if per_conflict_risk(c, p_d, delta_window=delta_window) > per_conflict_budget
    ]


@cache
def _binom_diff_at_offset(
    t_smaller: int,
    t_larger: int,
    p_d: float,
    signed_gap: int,
    delta_window: int = 0,
) -> float:
    """Compute ``Pr[|(k_l - k_s) + signed_gap| <= delta_window]`` exactly.

    With ``k_s ~ Binomial(t_smaller, p_d)`` and
    ``k_l ~ Binomial(t_larger, p_d)`` independent.

    *Collision-condition derivation.* The wall-clock arrival times
    are ``t_smaller + k_s`` and ``t_larger + k_l``. A realised
    collision within ``delta_window`` of coincidence requires
    ``|(t_smaller + k_s) - (t_larger + k_l)| <= delta_window``,
    which simplifies via ``signed_gap = t_larger - t_smaller >= 0``
    to ``|(k_l - k_s) + signed_gap| <= delta_window``.

    The function is **monotone-decreasing in ``signed_gap``**: a
    larger nominal separation makes the target offset harder for the
    delay difference to land within.
    """
    if signed_gap < 0:
        raise ValueError(f"signed_gap must be non-negative, got {signed_gap}")
    pmf_s = binom.pmf(range(t_smaller + 1), t_smaller, p_d)
    pmf_l = binom.pmf(range(t_larger + 1), t_larger, p_d)
    total = 0.0
    for k_s in range(t_smaller + 1):
        ps = float(pmf_s[k_s])
        if ps == 0.0:
            continue
        for k_l in range(t_larger + 1):
            if abs((k_l - k_s) + signed_gap) <= delta_window:
                total += ps * float(pmf_l[k_l])
    return total
