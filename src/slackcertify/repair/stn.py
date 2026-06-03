"""Provably-terminating wait-insertion repair via a Simple Temporal Network.

This is an *additive* alternative to :func:`slackcertify.repair.algorithm.slack_certify`
(the greedy repair that oscillates). It does NOT replace it — the two are
A/B tested. A single STN construction serves BOTH certification modes:

* **bounded** (Δ-disjointness): a single solve of the STN at separation
  parameter ``s = delta``. Every shared vertex / reverse edge is separated by
  a strict gap ``> delta`` (the required gap is ``delta + 1``).
* **probabilistic** (ε union bound under i.i.d. Bernoulli(``p_d``) per-step
  delays): a **binary search over a single global separation parameter**
  ``s``. For each candidate ``s`` the SAME STN that bounded mode builds for
  ``delta = s`` is solved and reconstructed, then the REAL verifier
  ``plan_risk_upper_bound(reconstructed, p_d, delta_window)`` is the in-loop
  acceptance test (``<= epsilon``). The smallest accepting ``s`` is returned.
  Because the actual verifier gates every acceptance, a returned plan can
  never fail :func:`is_epsilon_certified`; the total risk is monotone
  decreasing in ``s``, so the binary search is valid.

Method (Dechter, Meiri & Pearl 1991; the formulation MAPF-POST,
Hönig et al. ICAPS 2016 uses): the nominal plan fixes every agent's *cell
sequence*; only the *timing* may change, and only by ADDING waits. The
variables ``t[i, k]`` are the (repaired) timesteps at which agent ``i`` enters
the ``k``-th *compressed visit* of its fixed path, subject to:

* **Type-1 path-chain** ``t[i, k+1] >= t[i, k] + dwell_k`` — preserves motion
  order and the nominal dwell (so original waits are kept as lower bounds and
  never removed).
* **Source** ``t[i, 0] >= 0``.
* **Type-2 vertex separation** at separation parameter ``s``: consecutive-
  per-cell arcs (order fixed by nominal arrival, ties broken by ``agent_id``
  ascending — the TPG Type-2 tie-break) enforcing a strict gap ``> s``.
* **Swap/edge separation** for every reverse-edge pair the topology admits.
* **Goal occupancy** via a horizon node, so a separation requiring a
  *resting* agent to be the earlier occupant is infeasible (positive cycle).

Unified weights for separation parameter ``s`` (the required separation gap is
``s + 1``): transient vertex (source = earlier *exit* node) ``s``; goal-resting
vertex (source = horizon) ``s + 1``; swap (source = earlier *arrival* node)
``s + 1``.

A conflict-free nominal plan makes every arc point from a smaller nominal time
to a larger one, so the graph is a DAG and minimal entry times are longest-
path lengths from the source (topological DP, O(V+E)). When the required gaps
are inconsistent (a positive cycle), there is no fixed-order wait-only repair
at that ``s`` — the Bellman-Ford fallback detects it. :class:`STNInfeasible`
distinguishes (a) a bounded positive cycle (``reason='positive_cycle'``) from
(b) probabilistic ``s``-search exhaustion (``reason='s_max_exhausted'``).

This module does NOT self-trust in the hot path beyond the probabilistic
acceptance test (which is the real verifier). A debug-only re-check is
available behind the ``STN_DEBUG_VERIFY`` environment variable.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from slackcertify.certify.probabilistic import plan_risk_upper_bound
from slackcertify.core.graph import Cell
from slackcertify.core.plan import Path, Plan
from slackcertify.repair.algorithm import CertificationFailure

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["STNInfeasible", "stn_certify"]


# Sentinel STN nodes. Agent nodes are ``(agent_id: int, visit_index: int)``
# tuples; these string sentinels never collide with them.
_SOURCE = "__SOURCE__"
_HORIZON = "__HORIZON__"

# A compressed visit: (cell, nominal_entry_time, run_length).
_Visit = tuple[Cell, int, int]
_NEG = float("-inf")

# Hard cap on the probabilistic separation search.
_S_MAX_CAP = 64


class STNInfeasible(CertificationFailure):
    """No fixed-order, wait-only repair exists.

    ``reason`` distinguishes ``'positive_cycle'`` (a bounded solve at the
    requested ``delta`` has a positive cycle) from ``'s_max_exhausted'`` (the
    probabilistic search could not reach ``risk <= epsilon`` within the
    feasible / capped separation range). :attr:`cycle` carries one offending
    cycle when known (positive-cycle case).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Literal["positive_cycle", "s_max_exhausted"] = "positive_cycle",
        cycle: list[object] | None = None,
    ) -> None:
        """Attach the failure reason and (if known) the offending cycle."""
        super().__init__(message)
        self.reason: str = reason
        self.cycle: list[object] = list(cycle) if cycle is not None else []


def stn_certify(
    plan: Plan,
    delta: int = 0,
    *,
    mode: Literal["bounded", "probabilistic"] = "bounded",
    p_d: float | None = None,
    epsilon: float | None = None,
    return_diagnostics: bool = False,
) -> Plan | tuple[Plan, dict[str, object]]:
    """Insert waits (single pass / single search) so the mode's predicate holds.

    Parameters
    ----------
    plan
        A conflict-free nominal plan. Each agent's cell sequence is fixed;
        only timing changes, and only by adding waits.
    delta
        In ``mode='bounded'`` the Δ-disjointness margin (separation gap
        ``> delta``). In ``mode='probabilistic'`` the Δ-window of the
        realised-collision predicate (``0`` = point collision).
    mode
        ``'bounded'`` (single solve) or ``'probabilistic'`` (binary search on
        the global separation parameter ``s``, verifier-gated).
    p_d, epsilon
        Required for ``mode='probabilistic'``: per-step delay probability and
        the plan-level risk budget. Forbidden for ``mode='bounded'``.
    return_diagnostics
        When ``True`` return ``(plan, diagnostics)`` instead of just the plan.

    Returns
    -------
    Plan or (Plan, dict)
        The repaired plan, optionally with a diagnostics dict.

    Raises
    ------
    ValueError
        On a bad ``delta`` / ``mode`` / ``p_d`` / ``epsilon`` combination.
    STNInfeasible
        Bounded positive cycle, or probabilistic search exhaustion.
    """
    _validate_args(mode, delta, p_d, epsilon)

    path_by_id = {p.agent_id: p for p in plan.paths}
    agent_ids = sorted(path_by_id)
    visits: dict[int, list[_Visit]] = {
        aid: _compress(path_by_id[aid].vertices) for aid in agent_ids
    }

    if mode == "bounded":
        dist, num_vertex_arcs, num_swap_arcs, solver, cycle = _solve_stn(
            visits, agent_ids, delta
        )
        if cycle is not None:
            raise STNInfeasible(
                f"no fixed-order wait-only bounded repair at delta={delta} "
                f"(positive cycle of length {len(cycle)} in the STN)",
                reason="positive_cycle",
                cycle=cycle,
            )
        assert dist is not None
        repaired, total_waits = _reconstruct(plan, visits, agent_ids, dist)

        if os.environ.get("STN_DEBUG_VERIFY"):
            from slackcertify.certify.bounded import unsafe_pairs_bounded

            residual = unsafe_pairs_bounded(repaired, delta)
            assert not residual, f"STN_DEBUG_VERIFY: {len(residual)} residual conflict(s)"

        if return_diagnostics:
            diagnostics: dict[str, object] = {
                "mode": "bounded",
                "separation_s": delta,
                "total_waits": total_waits,
                "num_vertex_arcs": num_vertex_arcs,
                "num_swap_arcs": num_swap_arcs,
                "c_topo": num_vertex_arcs + num_swap_arcs,
                "solver": solver,
            }
            return repaired, diagnostics
        return repaired

    # --- probabilistic: binary search the global separation parameter s ---
    assert p_d is not None and epsilon is not None
    delta_window = delta
    chosen_s, repaired, risk, total_waits, nv, ns, solver, iters = _probabilistic_search(
        plan, visits, agent_ids, float(p_d), float(epsilon), delta_window
    )

    if os.environ.get("STN_DEBUG_VERIFY"):
        from slackcertify.certify.probabilistic import is_epsilon_certified

        assert is_epsilon_certified(
            repaired, float(p_d), float(epsilon), delta_window=delta_window
        ), "STN_DEBUG_VERIFY: epsilon certificate failed"

    if return_diagnostics:
        prob_diag: dict[str, object] = {
            "mode": "probabilistic",
            "separation_s": chosen_s,
            "total_waits": total_waits,
            "risk_ub": risk,
            "num_vertex_arcs": nv,
            "num_swap_arcs": ns,
            "c_topo": nv + ns,
            "solver": solver,
            "search_iters": iters,
        }
        return repaired, prob_diag
    return repaired


# --------------------------------------------------------------- internals


def _validate_args(
    mode: str, delta: int, p_d: float | None, epsilon: float | None
) -> None:
    """Reject mode / delta / p_d / epsilon combinations the STN can't honour."""
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    if mode == "bounded":
        if p_d is not None or epsilon is not None:
            raise ValueError("mode='bounded' must not set p_d or epsilon")
    elif mode == "probabilistic":
        if p_d is None or epsilon is None:
            raise ValueError("mode='probabilistic' requires p_d and epsilon")
        if not (0.0 <= p_d <= 1.0):
            raise ValueError(f"p_d must lie in [0, 1], got {p_d}")
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    else:
        raise ValueError(f"unsupported mode: {mode!r}")


def _solve_stn(
    visits: dict[int, list[_Visit]],
    agent_ids: list[int],
    s: int,
) -> tuple[dict[object, float] | None, int, int, str, list[object] | None]:
    """Build and solve the STN at separation parameter ``s`` (i.e. ``delta=s``).

    Returns ``(dist, num_vertex_arcs, num_swap_arcs, solver, cycle)``. ``dist``
    is ``None`` and ``cycle`` is a node list when the graph has a positive
    cycle (infeasible at this ``s``).
    """
    adj: dict[object, dict[object, int]] = defaultdict(dict)
    nodes: set[object] = {_SOURCE, _HORIZON}

    def add_arc(src: object, dst: object, w: int) -> None:
        nodes.add(src)
        nodes.add(dst)
        cur = adj[src].get(dst)
        if cur is None or w > cur:
            adj[src][dst] = w

    # Source + Type-1 chain + horizon arcs.
    for aid in agent_ids:
        vs = visits[aid]
        last = len(vs) - 1
        add_arc(_SOURCE, (aid, 0), 0)
        for k in range(last):
            add_arc((aid, k), (aid, k + 1), vs[k][2])  # dwell_k
        add_arc((aid, last), _HORIZON, 0)

    # Type-2 vertex separation, consecutive per shared cell (gap > s).
    events_by_cell: dict[Cell, list[tuple[int, int, int, bool]]] = defaultdict(list)
    for aid in agent_ids:
        vs = visits[aid]
        last = len(vs) - 1
        for k, (cell, nom_entry, _run) in enumerate(vs):
            events_by_cell[cell].append((nom_entry, aid, k, k == last))

    num_vertex_arcs = 0
    for events in events_by_cell.values():
        if len(events) < 2:
            continue
        events.sort(key=lambda e: (e[0], e[1]))  # (nominal_entry, agent_id)
        for a, b in zip(events, events[1:], strict=False):
            _, a_id, a_k, a_final = a
            _, b_id, b_k, _b_final = b
            if a_id == b_id:
                continue
            tgt = (b_id, b_k)
            if a_final:
                add_arc(_HORIZON, tgt, s + 1)  # resting earlier occupant
            else:
                add_arc((a_id, a_k + 1), tgt, s)  # transient: source = exit node
            num_vertex_arcs += 1

    # Swap/edge separation, all opposite-direction pairs per undirected edge.
    moves_by_dir: dict[tuple[Cell, Cell], list[tuple[int, int, int]]] = defaultdict(list)
    for aid in agent_ids:
        vs = visits[aid]
        for k in range(1, len(vs)):
            u = vs[k - 1][0]
            v = vs[k][0]
            moves_by_dir[(u, v)].append((vs[k][1] - 1, aid, k))  # nominal traversal start

    num_swap_arcs = 0
    seen_edges: set[tuple[Cell, Cell]] = set()
    for (u, v) in list(moves_by_dir):
        edge = (u, v) if u <= v else (v, u)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        fwd = moves_by_dir.get(edge, [])
        bwd = moves_by_dir.get((edge[1], edge[0]), [])
        for f_start, f_id, f_k in fwd:
            for b_start, b_id, b_k in bwd:
                if f_id == b_id:
                    continue
                if (f_start, f_id) <= (b_start, b_id):
                    earlier_node, later_node = (f_id, f_k), (b_id, b_k)
                else:
                    earlier_node, later_node = (b_id, b_k), (f_id, f_k)
                add_arc(earlier_node, later_node, s + 1)
                num_swap_arcs += 1

    order = _topo_order(nodes, adj)
    if order is not None:
        return _longest_path_dag(order, adj), num_vertex_arcs, num_swap_arcs, "topo", None
    dist, cycle = _bellman_ford_longest(nodes, adj, _SOURCE)
    if cycle is not None:
        return None, num_vertex_arcs, num_swap_arcs, "bellman_ford", cycle
    return dist, num_vertex_arcs, num_swap_arcs, "bellman_ford", None


def _reconstruct(
    plan: Plan,
    visits: dict[int, list[_Visit]],
    agent_ids: list[int],
    dist: dict[object, float],
) -> tuple[Plan, int]:
    """Expand computed entry times into repeated-vertex waits; return (plan, waits)."""
    total_waits = 0
    new_paths: list[Path] = []
    for aid in agent_ids:
        vs = visits[aid]
        last = len(vs) - 1
        verts: list[Cell] = []
        for k in range(last):
            dwell = int(dist[(aid, k + 1)]) - int(dist[(aid, k)])
            verts.extend([vs[k][0]] * dwell)  # chain arc guarantees dwell >= run_len >= 1
        verts.append(vs[last][0])  # final goal cell (verifier pads to horizon)
        new_paths.append(Path(agent_id=aid, vertices=verts))
        total_waits += int(dist[(aid, last)]) - vs[last][1]
    return Plan.from_paths(plan.agents, new_paths), total_waits


def _probabilistic_search(
    plan: Plan,
    visits: dict[int, list[_Visit]],
    agent_ids: list[int],
    p_d: float,
    epsilon: float,
    delta_window: int,
) -> tuple[int, Plan, float, int, int, str, int]:
    """Binary-search the smallest separation ``s`` with verifier risk ``<= epsilon``.

    The real verifier (:func:`plan_risk_upper_bound`) gates every acceptance,
    so the returned plan is guaranteed ε-certified. Total risk is monotone
    decreasing in ``s``; bounded feasibility is monotone decreasing in ``s``
    (a positive cycle at ``s`` persists at larger ``s``). ``s = 0`` is always
    feasible for a conflict-free nominal. Raises :class:`STNInfeasible`
    (``reason='s_max_exhausted'``) when no feasible ``s`` reaches the budget.
    """
    # memo[s] = (kind, plan|None, risk|None, total_waits, nv, ns, solver)
    memo: dict[int, tuple[str, Plan | None, float | None, int, int, int, str]] = {}

    def state(s: int) -> tuple[str, Plan | None, float | None, int, int, int, str]:
        cached = memo.get(s)
        if cached is not None:
            return cached
        dist, nv, ns, solver, _cycle = _solve_stn(visits, agent_ids, s)
        if dist is None:
            res: tuple[str, Plan | None, float | None, int, int, int, str] = (
                "cycle", None, None, 0, nv, ns, solver,
            )
        else:
            rep, total = _reconstruct(plan, visits, agent_ids, dist)
            risk = plan_risk_upper_bound(rep, p_d, delta_window=delta_window)
            kind = "ok" if risk <= epsilon else "high"
            res = (kind, rep, risk, total, nv, ns, solver)
        memo[s] = res
        return res

    # Exponential bracket: find an 'ok' upper bound, or the cycle boundary.
    hi: int | None = None
    cyc_at: int | None = None
    s = 1
    while True:
        if s > _S_MAX_CAP:
            raise STNInfeasible(
                f"probabilistic: epsilon={epsilon} unachievable within s_max="
                f"{_S_MAX_CAP} (union-bound risk still exceeds epsilon at s={_S_MAX_CAP})",
                reason="s_max_exhausted",
            )
        kind = state(s)[0]
        if kind == "ok":
            hi = s
            break
        if kind == "cycle":
            cyc_at = s
            break
        s *= 2  # 'high': need more separation

    if hi is not None:
        # [0, hi] all feasible; find the smallest 'ok' (risk monotone).
        chosen = _smallest_ok(state, 0, hi)
    else:
        assert cyc_at is not None
        # Feasibility lost at cyc_at; locate the top feasible s, then test it.
        s_feas = _largest_feasible(state, cyc_at)
        if state(s_feas)[0] == "ok":
            chosen = _smallest_ok(state, 0, s_feas)
        else:
            risk_at = state(s_feas)[2]
            raise STNInfeasible(
                f"probabilistic: epsilon={epsilon} unachievable by wait-only repair; "
                f"max feasible separation s={s_feas} yields risk {risk_at:.4g} > epsilon "
                f"(separation s={s_feas + 1} induces a positive cycle)",
                reason="s_max_exhausted",
            )

    kind, rep, risk, total, nv, ns, solver = state(chosen)
    assert rep is not None and risk is not None
    return chosen, rep, risk, total, nv, ns, solver, len(memo)


def _smallest_ok(state, lo: int, hi: int) -> int:  # noqa: ANN001 - local closure type
    """Smallest ``s`` in ``[lo, hi]`` with ``state(s) == 'ok'`` (all feasible)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if state(mid)[0] == "ok":
            hi = mid
        else:
            lo = mid + 1
    return lo


def _largest_feasible(state, cyc_at: int) -> int:  # noqa: ANN001 - local closure type
    """Largest ``s`` in ``[0, cyc_at)`` whose solve is not a positive cycle."""
    lo = cyc_at // 2  # feasible ('high' during doubling, or 0 which is always feasible)
    hi = cyc_at  # infeasible
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if state(mid)[0] == "cycle":
            hi = mid
        else:
            lo = mid
    return lo


def _compress(vertices: Sequence[Cell]) -> list[_Visit]:
    """Collapse maximal runs of an identical cell into ``(cell, entry, run_len)``."""
    out: list[_Visit] = []
    i = 0
    n = len(vertices)
    while i < n:
        cell = vertices[i]
        j = i
        while j < n and vertices[j] == cell:
            j += 1
        out.append((cell, i, j - i))
        i = j
    return out


def _topo_order(
    nodes: set[object], adj: dict[object, dict[object, int]]
) -> list[object] | None:
    """Return a topological order (Kahn) or ``None`` if a cycle exists."""
    indeg: dict[object, int] = dict.fromkeys(nodes, 0)
    for succ in adj.values():
        for dst in succ:
            indeg[dst] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    order: list[object] = []
    while queue:
        n = queue.pop()
        order.append(n)
        for dst in adj.get(n, {}):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)
    if len(order) != len(nodes):
        return None
    return order


def _longest_path_dag(
    order: list[object], adj: dict[object, dict[object, int]]
) -> dict[object, float]:
    """Longest-path lengths from ``_SOURCE`` over a DAG given a topo ``order``."""
    dist: dict[object, float] = dict.fromkeys(order, _NEG)
    dist[_SOURCE] = 0.0
    for n in order:
        if dist[n] == _NEG:
            continue
        base = dist[n]
        for dst, w in adj.get(n, {}).items():
            cand = base + w
            if cand > dist[dst]:
                dist[dst] = cand
    return dist


def _bellman_ford_longest(
    nodes: set[object],
    adj: dict[object, dict[object, int]],
    source: object,
) -> tuple[dict[object, float], list[object] | None]:
    """Longest paths from ``source`` with positive-cycle detection.

    Returns ``(dist, cycle)``. ``cycle`` is ``None`` when no positive cycle is
    reachable; otherwise it is one offending cycle (list of nodes).
    """
    edges = [(u, v, w) for u, succ in adj.items() for v, w in succ.items()]
    dist: dict[object, float] = dict.fromkeys(nodes, _NEG)
    pred: dict[object, object] = {}
    dist[source] = 0.0
    n = len(nodes)
    for _ in range(max(n - 1, 0)):
        updated = False
        for u, v, w in edges:
            du = dist[u]
            if du != _NEG and du + w > dist[v]:
                dist[v] = du + w
                pred[v] = u
                updated = True
        if not updated:
            break

    relaxable: object | None = None
    for u, v, w in edges:
        du = dist[u]
        if du != _NEG and du + w > dist[v]:
            relaxable = v
            pred[v] = u
            break
    if relaxable is None:
        return dist, None

    x = relaxable
    for _ in range(n):
        x = pred.get(x, x)
    cycle: list[object] = [x]
    cur = pred.get(x, x)
    while cur != x and cur not in cycle:
        cycle.append(cur)
        cur = pred.get(cur, cur)
    cycle.append(x)
    cycle.reverse()
    return dist, cycle
