"""Slack-certifier driver — Algorithm 1 of the ASYU 2026 paper.

The :func:`slack_certify` entry point loops at most ``max_outer_rounds``
times: each round detects all unsafe pairs, picks one according to the
chosen ordering, computes the minimum wait insertion ``w`` from Eq. (1),
applies it via :func:`propagate_shift` (or :func:`resolve_edge_conflict`),
then verifies the temporal-plan-graph is still acyclic. The loop exits
as soon as no unsafe pair remains; if the budget cannot be satisfied
:class:`BudgetInfeasibleError` is raised.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
>>> certified, cert = slack_certify(Plan.from_paths(a, p), mode="bounded", delta=0)
>>> cert.mode, cert.total_wait_inserted
('bounded', 0)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np

from slackcertify.certify.bounded import unsafe_pairs_bounded
from slackcertify.certify.certificate import Certificate
from slackcertify.certify.probabilistic import (
    _binom_diff_at_offset,
    per_conflict_risk,
)
from slackcertify.core.conflict import (
    Conflict,
    EdgeConflict,
    VertexConflict,
    detect_conflicts,
)
from slackcertify.core.plan import Plan
from slackcertify.core.tpg import build_tpg
from slackcertify.repair.budget import risk_proportional_budget, uniform_budget
from slackcertify.repair.deadlock import check_acyclicity_invariant
from slackcertify.repair.edge_handler import resolve_edge_conflict
from slackcertify.repair.ordering import (
    random_order,
    risk_descending_order,
    topological_order,
)
from slackcertify.repair.propagation import propagate_shift
from slackcertify.repair.wait_feasibility import is_wait_resolvable

__all__ = [
    "BudgetInfeasibleError",
    "CertificationFailure",
    "WaitInfeasibleError",
    "slack_certify",
]


class CertificationFailure(Exception):  # noqa: N818 - public API uses this name
    """The slack certifier could not produce a valid certificate."""


class BudgetInfeasibleError(CertificationFailure):
    """Probabilistic ε-budget cannot be met within the search bound."""


class WaitInfeasibleError(CertificationFailure):
    """Wait insertion alone cannot resolve the plan's conflict set.

    Raised by :func:`slack_certify` when the structural pre-check
    (:func:`slackcertify.repair.wait_feasibility.is_wait_resolvable`)
    detects a head-on swap on a topology where both agents traverse
    exactly the same cell set — no number of waits can make them pass.
    The :attr:`unresolvable_conflicts` attribute carries the offending
    :class:`EdgeConflict` instances so callers can decide whether to
    re-route or surface the issue to the user.
    """

    def __init__(self, message: str, unresolvable_conflicts: list[Conflict]) -> None:
        """Attach the unresolvable-conflict list to the exception."""
        super().__init__(message)
        self.unresolvable_conflicts: list[Conflict] = list(unresolvable_conflicts)


_PROBABILISTIC_MAX_W: int = 4096


def slack_certify(
    plan: Plan,
    mode: Literal["bounded", "probabilistic"],
    delta: int | None = None,
    p_d: float | None = None,
    epsilon: float | None = None,
    ordering: Literal["topological", "risk", "random"] = "topological",
    budget_alloc: Literal["uniform", "risk_proportional"] = "uniform",
    max_outer_rounds: int | None = None,
    rng: np.random.Generator | None = None,
    solver_used: str = "external",
) -> tuple[Plan, Certificate]:
    """Slack-certify ``plan`` under the chosen mode.

    Parameters
    ----------
    plan
        Input plan (from any MAPF solver).
    mode
        Either ``'bounded'`` (Δ-disjointness) or ``'probabilistic'``
        (ε-bound under i.i.d. Bernoulli per-step delays).
    delta, p_d, epsilon
        Mode-specific parameters. ``mode='bounded'`` requires ``delta``
        (the Δ-disjointness margin) and forbids ``p_d`` / ``epsilon``.
        ``mode='probabilistic'`` requires ``p_d`` and ``epsilon`` and
        accepts ``delta`` *optionally* as the **Δ-window** for the
        realised-collision predicate
        ``|D_j(t_j) - D_i(t_i) - (t_j - t_i)| <= delta``;
        when omitted (the default) it is treated as 0, the
        point-collision form. ``delta`` therefore serves a single,
        unified role across both modes — the Δ-margin of arrival-time
        coincidence — matching §III Definition 1.
    ordering
        Conflict-picking strategy passed through to the ordering helpers.
    budget_alloc
        Per-conflict budget allocation in probabilistic mode.
    max_outer_rounds
        Hard cap on the number of outer iterations. Defaults to the
        number of agents.
    rng
        Random generator used by the ``"random"`` ordering.
    solver_used
        Free-form label that ends up in the certificate.

    Returns
    -------
    (Plan, Certificate)
        The slack-certified plan and the matching certificate.

    Raises
    ------
    CertificationFailure
        If the algorithm cannot certify within ``max_outer_rounds``.
    BudgetInfeasibleError
        If the probabilistic per-conflict budget cannot be met within
        the internal wait-insertion search bound.
    """
    _validate_args(mode, delta, p_d, epsilon)
    if max_outer_rounds is None:
        max_outer_rounds = max(1, len(plan.agents))
    if rng is None:
        rng = np.random.default_rng(0)

    # Wait-feasibility precheck. A head-on swap on a topology where both
    # agents traverse exactly the same cell set cannot be resolved by
    # waits alone, no matter how many outer rounds we run. Catch this
    # upfront with an actionable diagnostic instead of letting the loop
    # fail with the generic "did not converge" message.
    initial_unsafe = _unsafe_pairs(plan, mode, delta, p_d, epsilon, budget_alloc)
    feasible, unresolvable = is_wait_resolvable(plan, initial_unsafe)
    if not feasible:
        details = "\n".join(f"  - {c}" for c in unresolvable)
        raise WaitInfeasibleError(
            f"wait insertion cannot resolve {len(unresolvable)} structural "
            f"conflict(s) on the input plan's topology:\n{details}",
            unresolvable,
        )

    current = plan
    total_wait = 0
    per_conflict_waits: dict[str, int] = {}

    # |C_0| and |C_cum| are *instrumentation only* (the RQ5 budget-slack
    # ratio); neither feeds the probabilistic budget. |C_0| is the size of
    # the initial unsafe set against the input plan, reused from the
    # wait-feasibility precheck; |C_cum| accumulates |C_k| across every
    # completed outer round.
    #
    # Probabilistic-budget denominator note: the per-conflict ε-budget is
    # ε / |C^(k)|, computed against the *current* plan each round (see
    # uniform_budget(epsilon, len(detect_conflicts(...))) in _compute_w /
    # _unsafe_pairs) — i.e. per-round |C|, NOT the pinned |C_0|. This stays
    # consistent with the ε-certificate verifier (is_epsilon_certified),
    # which checks the raw union-bound Σ_c P_c ≤ ε with no per-conflict
    # denominator: at convergence every surviving pair has P_c ≤ ε/|C^(k)|,
    # so the union bound is ≤ |C^(k)| · ε/|C^(k)| = ε. Algorithm and
    # verifier therefore agree and the emitted certificate is sound.
    c_0 = len(initial_unsafe)
    c_cum = 0

    # Batch outer loop. Each round detects every unsafe pair against the
    # current plan once, processes the whole set in one pass, and then
    # re-detects on the next iteration. Within a round, conflicts already
    # resolved by an earlier shift are skipped via _is_still_unsafe.
    rounds_completed = 0
    for _round in range(max_outer_rounds):
        unsafe = _unsafe_pairs(current, mode, delta, p_d, epsilon, budget_alloc)
        c_cum += len(unsafe)
        if not unsafe:
            return current, _build_certificate(
                current,
                mode,
                delta,
                p_d,
                epsilon,
                total_wait,
                per_conflict_waits,
                solver_used,
                ordering,
                budget_alloc,
                rounds_completed,
                c_0,
                c_cum,
            )

        ordered = _order(unsafe, current, ordering, p_d, rng)
        for c in ordered:
            if not _is_still_unsafe(c, current, mode, delta, p_d, epsilon, budget_alloc):
                continue
            w = _compute_w(c, mode, delta, p_d, epsilon, current, budget_alloc)
            if w == 0:
                # An earlier shift in this round already pushed this pair
                # outside the unsafe window; nothing to do.
                continue

            current = _apply_repair(current, c, w, mode, delta, p_d, epsilon, budget_alloc)
            check_acyclicity_invariant(build_tpg(current))

            total_wait += w
            per_conflict_waits[_conflict_key(c)] = per_conflict_waits.get(_conflict_key(c), 0) + w

        rounds_completed = _round + 1

    # After `max_outer_rounds` batches the loop body may have driven
    # `unsafe` to the empty set on its final pass — do one more detection
    # here so a tight `max_outer_rounds = n_agents` cap (Lemma 1)
    # succeeds when the batch loop actually converged on its last round.
    remaining = _unsafe_pairs(current, mode, delta, p_d, epsilon, budget_alloc)
    if not remaining:
        return current, _build_certificate(
            current,
            mode,
            delta,
            p_d,
            epsilon,
            total_wait,
            per_conflict_waits,
            solver_used,
            ordering,
            budget_alloc,
            rounds_completed,
            c_0,
            c_cum,
        )
    raise CertificationFailure(
        f"slack_certify did not converge in {max_outer_rounds} outer rounds; "
        f"{len(remaining)} unsafe pair(s) remain"
    )


# ---------------------------------------------------------------- internals


def _validate_args(mode: str, delta: int | None, p_d: float | None, epsilon: float | None) -> None:
    """Reject mode / delta / p_d / epsilon combinations that the certifier can't honour."""
    if mode == "bounded":
        if delta is None:
            raise ValueError("mode='bounded' requires delta")
        if delta < 0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if p_d is not None or epsilon is not None:
            raise ValueError("mode='bounded' must not set p_d or epsilon")
    elif mode == "probabilistic":
        if p_d is None or epsilon is None:
            raise ValueError("mode='probabilistic' requires p_d and epsilon")
        if not (0.0 <= p_d <= 1.0):
            raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        # delta is optional in probabilistic mode (Δ-window for arrival
        # coincidence); None is treated as 0 (point collision).
        if delta is not None and delta < 0:
            raise ValueError(f"delta must be non-negative, got {delta}")
    else:
        raise ValueError(f"unsupported mode: {mode!r}")


def _unsafe_pairs(
    plan: Plan,
    mode: str,
    delta: int | None,
    p_d: float | None,
    epsilon: float | None,
    budget_alloc: str,
) -> list[Conflict]:
    """Return the conflicts that still violate the current mode's safety predicate."""
    if mode == "bounded":
        assert delta is not None
        return unsafe_pairs_bounded(plan, delta)
    assert p_d is not None and epsilon is not None
    delta_window = delta if delta is not None else 0
    pairs = detect_conflicts(plan, delta=max(plan.makespan, 0))
    if not pairs:
        return []
    if budget_alloc == "uniform":
        share = uniform_budget(epsilon, len(pairs))
        return [c for c in pairs if per_conflict_risk(c, p_d, delta_window=delta_window) > share]
    budgets = risk_proportional_budget(epsilon, pairs, p_d)
    return [
        c
        for c in pairs
        if per_conflict_risk(c, p_d, delta_window=delta_window) > budgets.get(c, 0.0)
    ]


def _order(
    unsafe: list[Conflict],
    plan: Plan,
    ordering: str,
    p_d: float | None,
    rng: np.random.Generator,
) -> list[Conflict]:
    """Dispatch ``unsafe`` to the requested ordering strategy."""
    if ordering == "topological":
        return topological_order(unsafe, build_tpg(plan))
    if ordering == "risk":
        if p_d is None:
            raise ValueError("ordering='risk' requires probabilistic mode (p_d set)")
        return risk_descending_order(unsafe, p_d)
    if ordering == "random":
        return random_order(unsafe, rng)
    raise ValueError(f"unknown ordering: {ordering!r}")


def _compute_w(
    c: Conflict,
    mode: str,
    delta: int | None,
    p_d: float | None,
    epsilon: float | None,
    plan: Plan,
    budget_alloc: str,
) -> int:
    """Return the minimum wait count ``w`` that resolves conflict ``c``."""
    if mode == "bounded":
        assert delta is not None
        gap = abs(c.t_i - c.t_j)
        if gap > delta:
            return 0
        return delta - gap + 1
    assert p_d is not None and epsilon is not None
    delta_window = delta if delta is not None else 0
    pairs = detect_conflicts(plan, delta=max(plan.makespan, 0))
    if budget_alloc == "uniform":
        budget = uniform_budget(epsilon, len(pairs))
    else:
        budget = risk_proportional_budget(epsilon, pairs, p_d).get(c, 0.0)
    if c.t_i <= c.t_j:
        t_other, t_down = c.t_i, c.t_j
    else:
        t_other, t_down = c.t_j, c.t_i
    for w in range(_PROBABILISTIC_MAX_W + 1):
        new_t_down = t_down + w
        gap = abs(new_t_down - t_other)
        risk = _binom_diff_at_offset(
            t_smaller=min(new_t_down, t_other),
            t_larger=max(new_t_down, t_other),
            p_d=p_d,
            signed_gap=gap,
            delta_window=delta_window,
        )
        if risk <= budget:
            return w
    raise BudgetInfeasibleError(
        f"per-conflict budget {budget:.6g} unmeetable within "
        f"{_PROBABILISTIC_MAX_W} wait steps for {c}"
    )


def _is_still_unsafe(
    c: Conflict,
    plan: Plan,
    mode: str,
    delta: int | None,
    p_d: float | None,
    epsilon: float | None,
    budget_alloc: str,
) -> bool:
    """Return ``True`` iff ``c`` still represents an unsafe condition in ``plan``.

    Used by the batch outer loop to skip conflicts that an earlier shift
    in the same round has already resolved. The check is a *physical*
    one: the conflict's stored ``(agent, vertex, time_step)`` triples
    must still match what the (possibly mutated) plan reports, *and* the
    mode-specific unsafety predicate must still fire.
    """
    if isinstance(c, VertexConflict):
        if plan.vertex_visit(c.agent_i, c.t_i) != c.vertex:
            return False
        if plan.vertex_visit(c.agent_j, c.t_j) != c.vertex:
            return False
    elif isinstance(c, EdgeConflict):
        if (
            plan.vertex_visit(c.agent_i, c.t_i) != c.u
            or plan.vertex_visit(c.agent_i, c.t_i + 1) != c.v
            or plan.vertex_visit(c.agent_j, c.t_j) != c.v
            or plan.vertex_visit(c.agent_j, c.t_j + 1) != c.u
        ):
            return False

    if mode == "bounded":
        assert delta is not None
        return abs(c.t_i - c.t_j) <= delta

    # Probabilistic: re-check that this conflict's per-conflict risk
    # still exceeds its allocated budget under the current plan.
    assert p_d is not None and epsilon is not None
    delta_window = delta if delta is not None else 0
    pairs = detect_conflicts(plan, delta=max(plan.makespan, 0))
    if budget_alloc == "uniform":
        budget = uniform_budget(epsilon, len(pairs)) if pairs else float("inf")
    else:
        budget = risk_proportional_budget(epsilon, pairs, p_d).get(c, 0.0)
    return per_conflict_risk(c, p_d, delta_window=delta_window) > budget


def _apply_repair(
    plan: Plan,
    c: Conflict,
    w: int,
    mode: str,
    delta: int | None,
    p_d: float | None,
    epsilon: float | None,
    budget_alloc: str,
) -> Plan:
    """Apply ``w`` waits to resolve conflict ``c`` and return the shifted plan."""
    if isinstance(c, EdgeConflict):
        if mode == "bounded":
            return resolve_edge_conflict(plan, c, delta=delta, mode="bounded")
        pairs = detect_conflicts(plan, delta=max(plan.makespan, 0))
        if budget_alloc == "uniform":
            assert epsilon is not None
            budget = uniform_budget(epsilon, len(pairs))
        else:
            assert p_d is not None and epsilon is not None
            budget = risk_proportional_budget(epsilon, pairs, p_d).get(c, 0.0)
        delta_window = delta if delta is not None else 0
        return resolve_edge_conflict(
            plan,
            c,
            delta=None,
            mode="probabilistic",
            p_d=p_d,
            per_conflict_budget=budget,
            delta_window=delta_window,
        )

    assert isinstance(c, VertexConflict)
    # Shift the *later* agent so its arrival at the shared vertex is
    # pushed past the earlier one's departure window.
    if c.t_i < c.t_j or (c.t_i == c.t_j and c.agent_i < c.agent_j):
        downstream = c.agent_j
    else:
        downstream = c.agent_i
    return propagate_shift(plan, downstream, c.vertex, w)


def _conflict_key(c: Conflict) -> str:
    """Return a stable string identifier for ``c`` used in per-conflict wait ledgers."""
    if isinstance(c, VertexConflict):
        return f"V({c.agent_i},{c.agent_j},{c.vertex},{c.t_i},{c.t_j})"
    return f"E({c.agent_i},{c.agent_j},{c.u}->{c.v},{c.t_i},{c.t_j})"


def _build_certificate(
    plan: Plan,
    mode: str,
    delta: int | None,
    p_d: float | None,
    epsilon: float | None,
    total_wait: int,
    per_conflict_waits: dict[str, int],
    solver_used: str,
    ordering: str,
    budget_alloc: str,
    outer_rounds: int,
    initial_unsafe_count: int,
    cumulative_unsafe_count: int,
) -> Certificate:
    """Assemble the :class:`Certificate` returned to callers of :func:`slack_certify`."""
    sketch = (
        f"slack_certify in mode={mode!r}, ordering={ordering!r}, "
        f"budget_alloc={budget_alloc!r}; "
        + (
            f"Δ-disjointness verified by Theorem 1 with delta={delta}."
            if mode == "bounded"
            else (
                f"Union-bound risk under Bernoulli(p_d={p_d}) verified by "
                f"Proposition 2 against epsilon={epsilon}."
            )
        )
    )
    return Certificate(
        mode="bounded" if mode == "bounded" else "probabilistic",
        # delta is the Δ-margin in both modes (bounded disjointness in
        # bounded mode; arrival-coincidence window in probabilistic mode).
        delta=delta,
        p_d=p_d if mode == "probabilistic" else None,
        epsilon=epsilon if mode == "probabilistic" else None,
        total_wait_inserted=total_wait,
        outer_rounds=outer_rounds,
        initial_unsafe_count=initial_unsafe_count,
        cumulative_unsafe_count=cumulative_unsafe_count,
        per_conflict_waits=dict(per_conflict_waits),
        solver_used=solver_used,
        plan_hash=Certificate.hash_plan(plan),
        created_at=datetime.now(tz=timezone.utc),
        proof_sketch=sketch,
    )
