#!/usr/bin/env python3
"""A/B validation harness for stn_certify vs the greedy slack_certify.

Primary path: load ``warehouse-10-20-10-2-1`` (seeds 1..10, n in {20, 50})
via :class:`ScenarioRegistry` and solve with LaCAM*. If the MovingAI
benchmark is absent (this sandbox blocks the movingai.com host), the harness
falls back to *self-generated congested instances* — a synthetic warehouse-
like grid with 2x2 shelf blocks, solved by the same LaCAM* binary — so the
report is still produced and the real ``unsafe_pairs_bounded`` soundness gate
still runs. The fallback is loudly labelled; metrics are otherwise identical.

All correctness is measured EXTERNALLY here (via the existing verifier);
stn_certify is never trusted to self-certify.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

from slackcertify.benchmarks.scenarios import ScenarioRegistry
from slackcertify.certify.bounded import unsafe_pairs_bounded
from slackcertify.core.conflict import EdgeConflict, VertexConflict
from slackcertify.core.graph import Cell, GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.tpg import build_tpg
from slackcertify.repair.algorithm import CertificationFailure, slack_certify
from slackcertify.repair.stn import STNInfeasible, stn_certify
from slackcertify.solvers.base import SolverError
from slackcertify.solvers.lacam import LaCAMStarSolver

MAP_NAME = "warehouse-10-20-10-2-1"
SEEDS = list(range(1, 11))
AGENT_COUNTS = [20, 50]
DELTA = 2
TIME_LIMIT_S = 2.0
# The greedy oscillates and runs its full round cap on congested instances
# (~68s at n=20, far longer at n=50). Time-box it so the harness finishes;
# a greedy that cannot certify within the budget is recorded as not-certified
# (its raw outcome would be FAIL after 100 oscillating rounds anyway).
GREEDY_CAP_S = 20.0


# --------------------------------------------------------------- instances


def _synthetic_instance(n_agents: int, seed: int) -> tuple[GridGraph, list[Agent]]:
    """A congested warehouse-like grid: 2x2 shelf blocks with width-2 aisles."""
    import numpy as np

    width = height = 30
    obstacles: set[Cell] = {
        (x, y)
        for x in range(width)
        for y in range(height)
        if (x % 4 in (1, 2)) and (y % 4 in (1, 2))
    }
    free = [
        (x, y)
        for x in range(width)
        for y in range(height)
        if (x, y) not in obstacles
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(free)
    starts = free[:n_agents]
    goals = free[n_agents : 2 * n_agents]
    graph = GridGraph(width, height, obstacles)
    agents = [
        Agent(id=i, start=starts[i], goal=goals[i]) for i in range(n_agents)
    ]
    return graph, agents


def _load_instances() -> tuple[list[tuple[str, int, int, GridGraph, list[Agent]]], bool]:
    """Return (instances, synthetic?) — warehouse if available, else synthetic."""
    registry = ScenarioRegistry(root=Path("benchmarks"))
    try:
        registry.load_instance(MAP_NAME, AGENT_COUNTS[0], SEEDS[0])
        synthetic = False
    except FileNotFoundError:
        synthetic = True

    instances: list[tuple[str, int, int, GridGraph, list[Agent]]] = []
    for n in AGENT_COUNTS:
        for seed in SEEDS:
            if synthetic:
                graph, agents = _synthetic_instance(n, seed)
                label = f"synthWH/n{n}/s{seed}"
            else:
                graph, agents = registry.load_instance(MAP_NAME, n, seed)
                label = f"{MAP_NAME}/n{n}/s{seed}"
            instances.append((label, n, seed, graph, agents))
    return instances, synthetic


# --------------------------------------------------------------- evaluation


def _eval_stn(plan: Plan, delta: int) -> dict[str, object]:
    """Run stn_certify, then independently verify with unsafe_pairs_bounded."""
    t0 = time.perf_counter()
    infeasible = False
    cycle_len = 0
    repaired: Plan | None = None
    diag: dict[str, object] = {}
    try:
        result = stn_certify(plan, delta, return_diagnostics=True)
        assert isinstance(result, tuple)
        repaired, diag = result
    except STNInfeasible as exc:
        infeasible = True
        cycle_len = len(exc.cycle)
    wall = time.perf_counter() - t0

    residual_v = residual_e = 0
    acyclic: bool | None = None
    if repaired is not None:
        residual = unsafe_pairs_bounded(repaired, delta)
        residual_v = sum(isinstance(c, VertexConflict) for c in residual)
        residual_e = sum(isinstance(c, EdgeConflict) for c in residual)
        acyclic = build_tpg(repaired).is_acyclic()

    certified = (not infeasible) and residual_v == 0 and residual_e == 0 and acyclic is True
    return {
        "terminated": True,  # stn_certify has no cap; reaching here proves it returned
        "infeasible": infeasible,
        "cycle_len": cycle_len,
        "wall": wall,
        "total_waits": diag.get("total_waits", 0),
        "solver": diag.get("solver", "-"),
        "num_vertex_arcs": diag.get("num_vertex_arcs", 0),
        "num_swap_arcs": diag.get("num_swap_arcs", 0),
        "residual_v": residual_v,
        "residual_e": residual_e,
        "acyclic": acyclic,
        "certified": certified,
        "repaired": repaired,
    }


def _eval_old_greedy(plan: Plan, delta: int) -> str:
    """Run the OLD greedy (cap 100 rounds), time-boxed; return 'ok'|'fail'|'timeout'."""

    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, GREEDY_CAP_S)
    try:
        slack_certify(plan, "bounded", delta=delta, max_outer_rounds=100)
        return "ok"
    except CertificationFailure:
        return "fail"
    except TimeoutError:
        return "timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


# --------------------------------------------------------------- main


def main() -> None:
    instances, synthetic = _load_instances()
    solver = LaCAMStarSolver()

    if synthetic:
        print("=" * 100)
        print("!! WARNING: MovingAI host is blocked in this sandbox "
              "(benchmarks/maps is empty, fetch returns 403 'Host not in")
        print("!! allowlist'). Falling back to SELF-GENERATED congested instances "
              "(synthetic warehouse-like grid,")
        print("!! 2x2 shelves, solved by the real LaCAM* binary). "
              "All STN/verifier/greedy logic is identical; only the")
        print("!! instance source differs from warehouse-10-20-10-2-1.")
        print("=" * 100)

    rows: list[dict[str, object]] = []
    for label, n, seed, graph, agents in instances:
        try:
            plan = solver.solve(graph, agents, time_limit_s=TIME_LIMIT_S)
        except SolverError as exc:
            print(f"[skip] {label}: LaCAM* failed: {exc}")
            continue
        stn = _eval_stn(plan, DELTA)
        old_status = _eval_old_greedy(plan, DELTA)
        rows.append({"label": label, "n": n, "seed": seed, "old_status": old_status, **stn})

    # ---- (1) per-instance table ----
    print("\n" + "=" * 118)
    print(f"PER-INSTANCE (delta={DELTA})")
    print("-" * 118)
    hdr = (
        f"{'instance':30} {'OLD':>7} | {'term':>4} {'res_V':>5} {'res_E':>5} "
        f"{'acyc':>5} {'infeas':>6} {'waits':>6} {'solver':>13} {'wall_s':>8}"
    )
    print(hdr)
    print("-" * 118)
    for r in rows:
        acyc = "-" if r["acyclic"] is None else ("yes" if r["acyclic"] else "NO")
        old_disp = {"ok": "ok", "fail": "FAIL", "timeout": "TMO"}[str(r["old_status"])]
        print(
            f"{r['label']:30} {old_disp:>7} | "
            f"{('yes' if r['terminated'] else 'NO'):>4} "
            f"{r['residual_v']:>5} {r['residual_e']:>5} {acyc:>5} "
            f"{('YES' if r['infeasible'] else 'no'):>6} {r['total_waits']:>6} "
            f"{str(r['solver']):>13} {float(r['wall']):>8.4f}"
        )

    # ---- (2) aggregate ----
    n_total = len(rows)
    stn_cert = sum(1 for r in rows if r["certified"])
    stn_infeas = sum(1 for r in rows if r["infeasible"])
    old_cert = sum(1 for r in rows if r["old_status"] == "ok")
    old_timeout = sum(1 for r in rows if r["old_status"] == "timeout")
    all_term = all(r["terminated"] for r in rows)

    print("\n" + "=" * 118)
    print(f"AGGREGATE over N = {n_total} instances")
    print("-" * 118)
    print(f"  STN certified (residual==0 AND acyclic) : {stn_cert}/{n_total}")
    print(f"  STN reported infeasible                 : {stn_infeas}/{n_total}")
    print(f"  OLD greedy certified                    : {old_cert}/{n_total} "
          f"(of which {old_timeout} hit the {GREEDY_CAP_S:.0f}s time-box -> not certified)")
    if n_total:
        print(f"  STN certify-rate                        : {stn_cert / n_total:.1%}")
        print(f"  OLD certify-rate                        : {old_cert / n_total:.1%}")
    print(f"  stn_certify terminated on every instance: {all_term}")

    # ---- (3) fixed-order infeasibility rate (KEY DECISION NUMBER) ----
    print("\n" + "*" * 118)
    rate = (stn_infeas / n_total) if n_total else 0.0
    print(f"*** KEY DECISION NUMBER -- FIXED-ORDER INFEASIBILITY RATE: "
          f"{stn_infeas}/{n_total} = {rate:.1%}")
    print("*** (fraction of instances with NO fixed-order wait-only delta-repair)")
    print("*" * 118)

    # ---- (4) LOUD warnings: claimed success but residual!=0 or cyclic ----
    bugs = [
        r for r in rows
        if (not r["infeasible"])
        and (r["residual_v"] or r["residual_e"] or r["acyclic"] is not True)
    ]
    if bugs:
        print("\n" + "#" * 118)
        print(f"#### ENCODING BUG: {len(bugs)} instance(s) where STN returned a plan "
              "but the verifier DISAGREES (residual!=0 or TPG cyclic):")
        for r in bugs:
            print(f"####   {r['label']}: residual_V={r['residual_v']} "
                  f"residual_E={r['residual_e']} acyclic={r['acyclic']}")
        print("#" * 118)
    else:
        print("\n[OK] No encoding-bug instances: every STN plan that was returned "
              "verifies (residual==0) and is acyclic.")

    # ---- seed=1, n=20: delta sweep {1,2,3} ----
    print("\n" + "=" * 118)
    print("SEED=1 n=20 -- STN at delta in {1, 2, 3} (the case the greedy oscillates on):")
    print("-" * 118)
    target = next(
        ((lab, g, ag) for lab, n, s, g, ag in instances if n == 20 and s == 1), None
    )
    if target is None:
        print("  (instance unavailable)")
    else:
        lab, g, ag = target
        try:
            plan = solver.solve(g, ag, time_limit_s=TIME_LIMIT_S)
            for d in (1, 2, 3):
                res = _eval_stn(plan, d)
                old_status = _eval_old_greedy(plan, d)
                verdict = (
                    "INFEASIBLE" if res["infeasible"]
                    else ("CERTIFIED" if res["certified"] else "CLAIMED-BUT-RESIDUAL")
                )
                print(
                    f"  delta={d}: STN={verdict:>22} (res_V={res['residual_v']} "
                    f"res_E={res['residual_e']} acyclic={res['acyclic']} "
                    f"waits={res['total_waits']} solver={res['solver']}) | "
                    f"OLD greedy={old_status}"
                )
        except SolverError as exc:
            print(f"  LaCAM* failed: {exc}")


if __name__ == "__main__":
    main()
