# Formal Guarantees

This document is the canonical reference for the three formal claims
of `slack-certify-mapf`. Each claim is stated with its preconditions,
the formal statement, a proof sketch, the implementation reference,
and the property test that pins it. Cross-reference
[`docs/algorithm.md`](algorithm.md) for the derivations of the
underlying formulas (per-conflict-vs-batch outer loop, the
probabilistic risk formula, the simulator-alignment Δ-window).

## Theorem 1: Δ-Soundness (Bounded Mode)

**Preconditions.**
- A plan `π` whose paths are pairwise wait-resolvable
  (`is_wait_resolvable(π, conflicts(π, δ)) == True` —
  see `WaitInfeasibleError` below for the structural-impossibility
  case).
- A non-negative integer Δ.

**Statement.** A plan certified by
`slack_certify(plan, mode="bounded", delta=Δ)` is collision-free
under any per-step delay sequence with cumulative per-agent delay
≤ Δ.

**Proof sketch.** The certifier maintains the invariant that every
unsafe conflict pair `(i, j)` has `|t_i - t_j| > Δ` after wait
insertion. For any delay schedule with cumulative per-agent delay
≤ Δ, agent `i`'s realised arrival at the shared vertex lies in
`[t_i, t_i + Δ]` and agent `j`'s realised arrival lies in
`[t_j, t_j + Δ]`. Since `|t_i - t_j| > Δ`, the two intervals
`[t_i, t_i + Δ]` and `[t_j, t_j + Δ]` are disjoint, so the
realised arrivals cannot coincide and no collision occurs.

**Implementation reference.** `src/slackcertify/repair/algorithm.py`
(`slack_certify`, mode=`"bounded"` branch) +
`src/slackcertify/certify/bounded.py::is_delta_certified` (post-hoc
verifier).

**Property tests.**
[`tests/property/test_delta_soundness.py`](../tests/property/test_delta_soundness.py)
— three Hypothesis-driven tests pin the soundness across drawn
plans + Δ values:
- `test_certified_plan_survives_bounded_delays` (random delay
  schedules under the Δ budget).
- `test_certified_plan_survives_worst_case_adversary` (greedy
  adversary from `BoundedDelayModel.worst_case_against`).
- `test_zero_delta_is_no_op_for_already_safe_plans` (boundary).

## Lemma 1: Termination and Acyclicity

**Preconditions.**
- A plan `π` on which `slack_certify(plan, mode="bounded", delta=Δ)`
  is called.
- `max_outer_rounds ≥ n_agents` (the certifier raises
  `CertificationFailure` if the cap is too tight).

**Statement.** `slack_certify` terminates in at most `n_agents`
outer rounds; the TPG of the working plan remains acyclic
throughout.

**Proof sketch.** The outer loop processes conflicts in batches
keyed by a topological order on the round-starting TPG. A single
round consumes one row of the `|C|` ledger (each unsafe pair is
either resolved by its own wait insertion or by a downstream
shift in the same round, gated by `_is_still_unsafe`). Because
batches preserve the topological order, each round either
strictly reduces the unsafe pair count or terminates with the
empty set — bounding the round count by `n_agents`. Acyclicity
is preserved because wait insertions only add Type-1 edges
(per-agent ordering on the same path) and Type-2 edges from
already-acyclic ranks; no edge is added against the linear
extension.

**Implementation reference.** `src/slackcertify/repair/algorithm.py`
(`slack_certify`'s outer loop) +
`src/slackcertify/repair/deadlock.py::LinearExtensionTracker`
(maintains the rank function `σ` for the acyclicity check).

**Property tests.**
[`tests/property/test_acyclicity_invariant.py`](../tests/property/test_acyclicity_invariant.py)
— `test_certified_tpg_is_acyclic` (bounded mode),
`test_certified_tpg_acyclic_in_probabilistic_mode` (probabilistic
mode), `test_per_shift_tpg_acyclicity` (the
implementation-strengthened per-shift variant, described below).
[`tests/property/test_termination.py`](../tests/property/test_termination.py)
— `test_certifier_terminates_within_theoretical_bound`
(outer-round bound),
`test_certifier_at_default_bound_terminates` (default cap).

## Proposition 1: Probabilistic Union Bound

**Preconditions.**
- A plan `π`.
- A Bernoulli per-step delay rate `p_d ∈ [0, 1)`.
- A per-conflict safety budget `ε ∈ [0, 1]`.

**Statement.** For a plan certified under
`slack_certify(plan, mode="probabilistic", p_d=p_d, epsilon=ε)`
with per-conflict budget `ε / |C|` (uniform allocator) or its
risk-proportional analogue, the probability of any collision under
independent Bernoulli(`p_d`) per-step delays is at most `ε`.

**Proof sketch.** Let `C` be the set of pairwise vertex-/edge-
conflicts in `π`. By the union bound,
`Pr[some collision] ≤ ∑_{c ∈ C} Pr[collision at c]`. The
certifier enforces `Pr[collision at c] ≤ ε / |C|` for every
`c ∈ C` (or the equivalent budget under risk-proportional
allocation). Summing yields `Pr[some collision] ≤ ε`.

**Implementation note.** The per-conflict risk formula is

```
Pr[ |(D_j(t_j) − D_i(t_i)) + signed_gap| ≤ Δ ]
```

where `signed_gap = t_j − t_i` and `D_a(t_a) ~ Binomial(t_a, p_d)`
is the cumulative delay agent `a` has accumulated by its nominal
time of arrival at the shared vertex. The collision condition is
derived from the wall-clock arrival equation
`t_i + D_i(t_i) = t_j + D_j(t_j)` (or, with the Δ-margin, the
unsafe event being arrival times within Δ of each other). The
default `Δ = 0` form covers exact arrival-time coincidence; the
default-`Δ = 1` form used by the Phase 7 runners covers any
integer-tick overlap of occupation windows (see
[`docs/algorithm.md`](algorithm.md) §"Aligning with the simulator's
collision model" for the simulator-semantics justification).

**Implementation reference.** `src/slackcertify/certify/probabilistic.py`
(`per_conflict_risk`, `plan_risk_upper_bound`,
`is_epsilon_certified`) +
`src/slackcertify/repair/budget.py` (uniform / risk-proportional
allocators).

**Property tests.**
[`tests/property/test_probabilistic_bound.py`](../tests/property/test_probabilistic_bound.py)
— `test_empirical_rate_below_three_sigma_band` runs 1500
Bernoulli rollouts per drawn plan and asserts the empirical
realised-collision rate stays within 3σ of the certified ε.
[`tests/property/test_probabilistic_formula_invariants.py`](../tests/property/test_probabilistic_formula_invariants.py)
— pins the corrected formula in three directions
(monotone-decreasing in gap, the `p_d=0` boundary, the
`p_d`-low-range monotonicity that the buggy formula inverted).
[`tests/property/test_certifier_simulator_agreement.py`](../tests/property/test_certifier_simulator_agreement.py)
— `test_certifier_simulator_agreement` validates Proposition 1
against the **simulator** path
(`monte_carlo_rollout` + `Executor`) under `Δ = 1`, the
simulator-alignment convention; without `Δ = 1` the formula
under-estimates the simulator's realised-collision rate by
roughly `p_d`.

## Implementation-strengthened invariants

Two invariants are strictly stronger than the paper's claims.
They were discovered during development and are pinned here as
regression guards even though the paper does not need them.

### Per-shift TPG acyclicity (strengthens Lemma 1)

Lemma 1 claims **per-round** acyclicity (i.e., at the boundaries
of the outer loop). The implementation maintains **per-shift**
acyclicity: the TPG is acyclic after every individual
`_apply_repair` call, not just at round boundaries. This is a
useful operational invariant — it means a crash mid-round leaves
the working plan in a still-checkable state — but it is not a
claim the paper makes or relies on.

**Pinned by.**
- `tests/unit/test_repair_batch.py::test_intra_round_acyclicity`
  — exercises the invariant on a hand-built 4-agent hub fixture
  with detailed assertions on the topological order and the
  expected downstream-shift agents.
- `tests/property/test_acyclicity_invariant.py::test_per_shift_tpg_acyclicity`
  — Hypothesis-driven version that patches `_apply_repair` to
  capture every intermediate plan, then asserts acyclicity on
  every captured state. Exercises many more topologies than the
  single hub fixture above.

### WaitInfeasibleError (strengthens algorithmic completeness)

The certifier raises a structured `WaitInfeasibleError` on
corridor-swap topologies where no wait insertion can resolve the
conflict (both agents traverse exactly the same cell set in
opposite directions). This is operationally meaningful: it
distinguishes "ran out of rounds due to slow convergence"
(`CertificationFailure` without `unresolvable_conflicts`) from
"wait insertion is structurally impossible on this topology"
(`WaitInfeasibleError` with the offending `EdgeConflict`
instances attached as `exc.unresolvable_conflicts`).

The same precheck runs inside `solve_optimal_wait_insertion`
(the wait-insertion ILP) so the two methods agree on
infeasibility — this is the cross-method consistency check the
RQ4 runner uses.

**Pinned by.**
- `tests/property/test_wait_infeasibility.py::test_corridor_swap_raises_wait_infeasible[3..6]`
  — parametric tests across `n ∈ {3, 4, 5, 6}` agents in a
  corridor swap topology.
- `tests/unit/test_ilp_solver_small.py::test_ilp_rejects_corridor_swap_as_infeasible`
  — the ILP side of the same case, asserting
  `ILPResult.status == "infeasible"` with
  `diagnostics["reason"] == "wait_infeasible"`.
