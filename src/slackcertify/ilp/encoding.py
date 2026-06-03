"""ILP encoding of the optimal-wait-insertion problem.

Formulation overview
--------------------

Decision variables. For every agent ``i`` and every visit-index ``k``
in its nominal path, we introduce an integer wait variable
``w_{i,k} >= 0`` whose upper bound is ``delta * |same-cell pairs| + 1``
(a safe over-estimate of the maximum useful wait per visit). Inserting
``w_{i,k}`` waits at index ``k`` delays every subsequent vertex visit
by the same amount, so agent ``i``'s wall-clock arrival at its
nominal index ``t`` equals ``t + sum_{j < t} w_{i,j}``.

Objective. Minimise the total inserted waits ``sum_{i,k} w_{i,k}``.

Conflict constraints. **The encoding constrains every same-cell
*occupancy interval* pair across distinct agents — *not* just the
nominal conflict list, and *not* per nominal index.** Inserting waits
between two consecutive same-cell visits makes an agent occupy a cell
over a *contiguous interval* of realised ticks; a per-index separation
can hold at every endpoint yet still let another agent arrive *inside*
the interval (a real collision). So for every cell ``v`` we enumerate
each agent's **maximal contiguous run** of ``v`` (convention B: an agent
may leave and re-enter ``v``, giving several disjoint runs), and for
every cross-agent pair of runs at ``v`` we emit an *interval*
Δ-disjointness constraint:

* either agent ``A`` *leaves* ``v`` at least ``delta + 1`` ticks before
  ``B`` *enters*, or
* agent ``B`` leaves at least ``delta + 1`` ticks before ``A`` enters,

where ``enter`` / ``last`` are realised arrival times (linear in the
wait vars): ``enter = s + sum_{j<s} w``, last occupied tick
``last = e + sum_{j<=e} w`` for a run on indices ``[s, e]``. For an
agent's **terminal** run (it rests at its goal through the horizon)
``last = horizon`` — a safe upper bound on the realised makespan — which
forces every other agent to clear ``v`` *before* the goal-rester
settles. This matches the verifier exactly: it flags a vertex conflict
iff two occupancy ticks lie within ``delta``, so a gap of
``last_A + delta + 1 <= enter_B`` (i.e. closest ticks differ by
``> delta``) is precisely Δ-disjointness.

This guarantees soundness when wait shifts induce *new* conflicts that
weren't present in the nominal plan: the broader constraint set already
covers them. The nominal conflict list is no longer used to filter
constraints; it is retained only for diagnostics. (Edge conflicts are
subsumed: a head-on swap requires the two agents to coincide at *both*
endpoint cells, so interval-disjointness at either endpoint blocks it.)

We linearise each disjunction with a binary indicator
``y_{p} in {0, 1}`` (``1`` = the first listed visit arrives first)
and a Big-M constant ``M`` large enough to be a valid relaxation
slack (``M = sum_of_visits * w_ub + nominal_makespan + delta + 10``).

Global acyclicity (executability). Choosing passing orders freely via the
``y`` binaries lets the timing/disjointness constraints alone admit a
*globally cyclic* set of orders (``i`` before ``j`` at cell A, ``j`` before
``k`` at B, ``k`` before ``i`` at C) — locally consistent, globally a
deadlock, hence non-executable. Such a solution is only a lower bound. To
optimise over *executable* (acyclic-TPG) plans we add a continuous
**topological-position** variable ``pos[(agent, visit)]`` per agent-visit
node and tie the precedence to the *same* ``y``:

* When ``y`` selects ``u`` before ``w``: ``pos[u] + 1 <= pos[w]`` (active),
  the reverse relaxed by Big-M; and vice-versa when ``y`` selects ``w`` first.
* Type-1 path order (fixed, never flipped): ``pos[(a,k)] + 1 <= pos[(a,k+1)]``
  for each agent's own consecutive visits.

A consistent assignment of ``pos`` exists **iff** the chosen orders are
globally acyclic, so feasibility of ``pos`` certifies executability. The
position constraints are global (across all shared cells), not per-resource.
The objective is unchanged: still minimise total inserted waits.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> a = [Agent(id=0, start=(0, 0), goal=(2, 0)),
...      Agent(id=1, start=(2, 0), goal=(0, 0))]
>>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
...      Path(agent_id=1, vertices=[(2, 0), (1, 0), (0, 0)])]
>>> enc = build_wait_insertion_ilp(Plan.from_paths(a, p), delta=0)
>>> len(enc.wait_vars)
6
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import pulp
except ImportError as exc:  # pragma: no cover - exercised by users without [ilp]
    raise ImportError(
        "ILP support requires pulp. Install with: pip install 'slackcertify[ilp]'"
    ) from exc

from slackcertify.core.conflict import Conflict, detect_conflicts
from slackcertify.core.plan import Plan

__all__ = ["ILPEncoding", "build_wait_insertion_ilp"]


@dataclass(frozen=True)
class ILPEncoding:
    """Bundle of the LP problem, decision variables, and conflict list.

    ``pos_vars`` holds the topological-position variable for each
    ``(agent_id, visit_idx)`` node; their constraints enforce global
    acyclicity so any feasible solution is an executable (deadlock-free)
    plan.
    """

    problem: pulp.LpProblem
    wait_vars: dict[tuple[int, int], pulp.LpVariable]
    conflicts: list[Conflict]
    pos_vars: dict[tuple[int, int], pulp.LpVariable] = field(default_factory=dict)


def build_wait_insertion_ilp(
    plan: Plan,
    delta: int,
    conflicts: list[Conflict] | None = None,
    *,
    enforce_acyclicity: bool = True,
) -> ILPEncoding:
    """Build the wait-insertion ILP for ``plan`` at tolerance ``delta``.

    ``enforce_acyclicity`` (default ``True``) adds the global topological-
    position constraints that restrict feasible solutions to executable
    (deadlock-free) plans. Set it ``False`` only to reproduce the OLD,
    unconstrained lower-bound model for comparison (the optimum may then be a
    cyclic, non-executable set of passing orders).

    Parameters
    ----------
    plan
        Nominal MAPF plan (the wait-insertion is performed on its
        existing path topology, not by re-routing).
    delta
        Δ-disjointness tolerance; must be non-negative.
    conflicts
        Optional pre-computed unsafe-pairs list. Defaults to
        :func:`detect_conflicts(plan, delta)`.

    Returns
    -------
    ILPEncoding
        A bundle containing the unsolved :class:`pulp.LpProblem`, the
        decision variables indexed by ``(agent_id, visit_idx)``, and
        the list of conflicts the constraints encode.

    Raises
    ------
    ValueError
        If ``delta < 0``.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> enc = build_wait_insertion_ilp(Plan.from_paths(a, p), delta=0)
    >>> enc.conflicts
    []
    """
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")

    if conflicts is None:
        conflicts = detect_conflicts(plan, delta=delta)

    # Enumerate each agent's MAXIMAL CONTIGUOUS RUNS of an identical cell.
    # A run is (agent_id, cell, s, e, is_terminal): the agent occupies `cell`
    # over nominal indices [s, e] inclusive; `is_terminal` marks the final
    # goal-rest run (occupied through the horizon). Convention B: an agent may
    # leave and re-enter a cell, producing several disjoint runs at that cell.
    runs_by_cell: dict[tuple[int, int], list[tuple[int, int, int, bool]]] = defaultdict(list)
    for path in plan.paths:
        verts = path.vertices
        n = len(verts)
        k = 0
        while k < n:
            v = verts[k]
            e = k
            while e + 1 < n and verts[e + 1] == v:
                e += 1
            runs_by_cell[v].append((path.agent_id, k, e, e == n - 1))
            k = e + 1

    # Every cross-agent pair of runs at the same cell needs interval
    # disjointness. (cell, runA, runB) with stable canonical order.
    run_pairs: list[
        tuple[tuple[int, int, int, bool], tuple[int, int, int, bool], tuple[int, int]]
    ] = []
    for v, runs in runs_by_cell.items():
        for ra, rb in itertools.combinations(runs, 2):
            # ra/rb are (agent_id, s, e, is_terminal)
            if ra[0] == rb[0]:
                continue
            lo, hi = (ra, rb) if ra <= rb else (rb, ra)
            run_pairs.append((lo, hi, v))

    n_pairs = len(run_pairs)
    w_ub = delta * max(n_pairs, 1) + 1
    total_visits = sum(len(p.vertices) for p in plan.paths)
    # Safe upper bound on the realised makespan: nominal makespan plus the
    # largest possible total wait insertion. `horizon` is the `last` tick used
    # for terminal goal-rest runs; `big_m == horizon + delta + 10` is a valid
    # relaxation slack for the interval disjunction.
    horizon = total_visits * w_ub + plan.makespan
    big_m = horizon + delta + 10

    prob = pulp.LpProblem("slack_certify_wait_insertion", pulp.LpMinimize)

    wait_vars: dict[tuple[int, int], pulp.LpVariable] = {}
    pos_vars: dict[tuple[int, int], pulp.LpVariable] = {}
    for path in plan.paths:
        for k in range(len(path.vertices)):
            wait_vars[(path.agent_id, k)] = pulp.LpVariable(
                name=f"w_{path.agent_id}_{k}",
                lowBound=0,
                upBound=w_ub,
                cat="Integer",
            )
            if enforce_acyclicity:
                # Topological-position variable for the acyclicity
                # (executability) constraints. Continuous in [0, total_visits];
                # a feasible assignment exists iff the chosen passing orders are
                # acyclic.
                pos_vars[(path.agent_id, k)] = pulp.LpVariable(
                    name=f"pos_{path.agent_id}_{k}",
                    lowBound=0,
                    upBound=total_visits,
                    cat="Continuous",
                )

    prob += pulp.lpSum(wait_vars.values()), "total_waits"

    # Big-M for the position disjunction need only dominate the position range
    # (positions lie in [0, total_visits]); total_visits + 1 is a valid slack.
    pos_big_m = total_visits + 1

    # Type-1 path-order positions (fixed, never flipped): each agent's own
    # consecutive visits must increase in topological position.
    if enforce_acyclicity:
        for path in plan.paths:
            for k in range(len(path.vertices) - 1):
                prob += (
                    pos_vars[(path.agent_id, k)] + 1 <= pos_vars[(path.agent_id, k + 1)],
                    f"pathorder_{path.agent_id}_{k}",
                )

    for idx, (run_a, run_b, _v) in enumerate(run_pairs):
        _add_disjointness_constraint(
            prob,
            wait_vars,
            pos_vars if enforce_acyclicity else None,
            run_a=run_a,
            run_b=run_b,
            delta=delta,
            big_m=big_m,
            pos_big_m=pos_big_m,
            horizon=horizon,
            tag=f"p{idx}",
        )

    return ILPEncoding(
        problem=prob,
        wait_vars=wait_vars,
        conflicts=list(conflicts),
        pos_vars=pos_vars,
    )


def _add_disjointness_constraint(
    prob: pulp.LpProblem,
    wait_vars: dict[tuple[int, int], pulp.LpVariable],
    pos_vars: dict[tuple[int, int], pulp.LpVariable] | None,
    *,
    run_a: tuple[int, int, int, bool],
    run_b: tuple[int, int, int, bool],
    delta: int,
    big_m: int,
    pos_big_m: int,
    horizon: int,
    tag: str,
) -> None:
    """Append the interval Δ-disjointness constraint for two same-cell runs.

    ``run_* = (agent_id, s, e, is_terminal)`` denotes occupancy of the shared
    cell over nominal indices ``[s, e]``. Realised occupancy is the tick
    interval ``[enter, last]`` with ``enter = s + sum_{j<s} w`` and
    ``last = e + sum_{j<=e} w`` (or ``horizon`` for the terminal goal-rest
    run). The two intervals are Δ-disjoint iff one leaves at least ``delta+1``
    ticks before the other enters; the branch is selected by the binary ``y``
    that *also* drives the global-acyclicity precedence on the topological
    positions, so order, timing-interval, and acyclicity stay consistent.
    """
    a_i, s_i, e_i, term_i = run_a
    a_j, s_j, e_j, term_j = run_b

    enter_i = s_i + pulp.lpSum(wait_vars[(a_i, k)] for k in range(s_i))
    enter_j = s_j + pulp.lpSum(wait_vars[(a_j, k)] for k in range(s_j))
    last_i = horizon if term_i else (e_i + pulp.lpSum(wait_vars[(a_i, k)] for k in range(e_i + 1)))
    last_j = horizon if term_j else (e_j + pulp.lpSum(wait_vars[(a_j, k)] for k in range(e_j + 1)))

    y = pulp.LpVariable(name=f"y_{tag}_{a_i}_{a_j}", cat="Binary")
    # When y == 1: run A (run_a) finishes first; B enters >= delta+1 after A leaves.
    prob += (
        enter_j - last_i >= (delta + 1) - big_m * (1 - y),
        f"{tag}_a_first",
    )
    # When y == 0: run B finishes first; A enters >= delta+1 after B leaves.
    prob += (
        enter_i - last_j >= (delta + 1) - big_m * y,
        f"{tag}_b_first",
    )

    # Global acyclicity: the same y that picks the interval order picks the
    # precedence edge between the two run-head nodes u = (a_i, s_i),
    # w = (a_j, s_j). Skipped in the OLD unconstrained model (pos_vars is None).
    if pos_vars is None:
        return
    u = pos_vars[(a_i, s_i)]
    w = pos_vars[(a_j, s_j)]
    # When y == 1 (u before w): pos[u] + 1 <= pos[w]; reverse relaxed.
    prob += (
        u + 1 <= w + pos_big_m * (1 - y),
        f"{tag}_pos_u_before_w",
    )
    # When y == 0 (w before u): pos[w] + 1 <= pos[u]; this relaxed when y == 1.
    prob += (
        w + 1 <= u + pos_big_m * y,
        f"{tag}_pos_w_before_u",
    )
