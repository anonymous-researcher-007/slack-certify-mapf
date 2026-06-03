#!/usr/bin/env python3
"""Validate the interval-aware + acyclicity-constrained wait-insertion ILP (REAL data).

Real-data version of scripts/validate_ilp_acyclic.py: identical check logic, but
the instance source is the REAL MovingAI benchmark suite (loaded via
ScenarioRegistry) instead of synthetic instances, swept over both
warehouse-10-20-10-2-1 and random-32-32-20.

For each instance (map in {warehouse, random-32-32-20}, n in {4,8,12,16,20},
seeds 1-5, delta=2) this:
  (1) solves the fixed (interval-aware disjointness + global acyclicity) ILP,
  (2) realizes the solution as a Plan,
  (3) asserts detect_conflicts(plan, delta=2) residual == 0 (vertex AND swap)
      -- the same Delta-certificate stn_certify's output is checked against,
  (4) asserts build_tpg(plan).is_acyclic() == True,
  (5) cross-checks physical execution:
      monte_carlo_rollout(plan, BoundedDelayModel(delta=2), K=500).success_rate
      == 1.0.
Every SOLVED instance must pass (3),(4),(5). A failure means the interval
encoding is still wrong -- the offending (instance, check, residual cell/agents/
ticks) is reported; the test is NOT adjusted to pass.

Records per instance: ILP status (optimal / feasible-timeout / infeasible),
solve time, objective (total waits), and checks (3)(4)(5). The 3600s budget is
NOT lowered; n=16/20 may time out (reportable). Reports the largest n that
solves to proven optimality.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path as _P  # noqa: E402, N814

from slackcertify.benchmarks.scenarios import ScenarioRegistry  # noqa: E402
from slackcertify.certify.bounded import is_delta_certified  # noqa: E402
from slackcertify.core.conflict import EdgeConflict, VertexConflict, detect_conflicts  # noqa: E402
from slackcertify.core.plan import Plan  # noqa: E402
from slackcertify.core.tpg import build_tpg  # noqa: E402
from slackcertify.delay.bounded import BoundedDelayModel  # noqa: E402
from slackcertify.ilp.encoding import build_wait_insertion_ilp  # noqa: E402
from slackcertify.ilp.solver import (  # noqa: E402
    _augment_plan_with_waits,
    _make_backend,
    _resolve_solver,
)
from slackcertify.simulate.rollout import monte_carlo_rollout  # noqa: E402
from slackcertify.solvers.base import SolverError  # noqa: E402
from slackcertify.solvers.lacam import LaCAMStarSolver  # noqa: E402

try:
    import pulp
except ImportError:
    print("pulp not installed; run: pip install pulp")
    raise

N_AGENTS = [4, 8, 12, 16, 20]
SEEDS = [1, 2, 3, 4, 5]
DELTA = 2
TIME_LIMIT_S = 3600.0
ROLLOUTS = 500


def _solve(plan: Plan, time_limit_s: float):  # noqa: ANN202
    """Solve the fixed ILP; return (status, plan|None, obj, runtime)."""
    chosen = _resolve_solver("highs")
    enc = build_wait_insertion_ilp(plan, delta=DELTA)
    backend = _make_backend(chosen, time_limit_s=time_limit_s, msg=False)
    t0 = time.perf_counter()
    enc.problem.solve(backend)
    runtime = time.perf_counter() - t0
    obj = pulp.value(enc.problem.objective)
    has_feasible = obj is not None and obj == obj  # not NaN
    if enc.problem.status == pulp.LpStatusOptimal:
        status = "optimal"
    elif enc.problem.status == pulp.LpStatusInfeasible:
        status = "infeasible"
    elif runtime >= time_limit_s * 0.5:
        status = "feasible-timeout" if has_feasible else "timeout"
    else:
        status = pulp.LpStatus[enc.problem.status].lower()
    realized = _augment_plan_with_waits(plan, enc.wait_vars) if has_feasible else None
    return status, realized, (int(round(obj)) if has_feasible else None), runtime


def main() -> None:
    solver = LaCAMStarSolver()
    rows: list[dict[str, object]] = []
    n_solved = n_pass = 0
    largest_optimal = 0

    print("=" * 104)
    print(f"INTERVAL-AWARE + ACYCLIC ILP VALIDATION  (delta={DELTA}, budget={TIME_LIMIT_S:.0f}s)")
    print("real MovingAI instances: warehouse-10-20-10-2-1, random-32-32-20")
    print("=" * 104)
    print(f"{'instance':24} {'status':>16} {'time_s':>9} {'obj':>5} "
          f"{'(3)Δcert':>9} {'(4)acyc':>8} {'(5)rollout':>11} {'resV':>5} {'resE':>5}")
    print("-" * 104)

    _reg = ScenarioRegistry(root=_P("benchmarks"))
    for _map in ["warehouse-10-20-10-2-1", "random-32-32-20"]:
      for n in N_AGENTS:
        n_optimal_this = 0
        for seed in SEEDS:
            try:
                graph, agents = _reg.load_instance(_map, n, seed)
            except Exception as _e:  # noqa: BLE001
                print(f"{_map[:10]}/n{n}/s{seed}: load fail {str(_e)[:40]}")
                continue
            try:
                nominal = solver.solve(graph, agents, time_limit_s=2.0)
            except SolverError as exc:
                print(f"{_map[:10]}/n{n}/s{seed}: LaCAM* failed: {exc}")
                continue
            label = f"{_map[:10]}/n{n}/s{seed}"

            status, rep, obj, rt = _solve(nominal, time_limit_s=TIME_LIMIT_S)
            cert = acyc = None
            roll = float("nan")
            res_v = res_e = 0
            residual = []
            if rep is not None:
                n_solved += 1
                residual = detect_conflicts(rep, delta=DELTA)
                res_v = sum(isinstance(c, VertexConflict) for c in residual)
                res_e = sum(isinstance(c, EdgeConflict) for c in residual)
                cert = is_delta_certified(rep, DELTA)
                acyc = build_tpg(rep).is_acyclic()
                roll = monte_carlo_rollout(
                    rep, BoundedDelayModel(delta=DELTA), K=ROLLOUTS,
                    rng=np.random.default_rng(0),
                ).success_rate
                if cert and acyc and roll == 1.0:
                    n_pass += 1
            if status == "optimal":
                n_optimal_this += 1

            rows.append({
                "label": label, "status": status, "rt": rt, "obj": obj,
                "cert": cert, "acyc": acyc, "roll": roll,
                "res_v": res_v, "res_e": res_e, "residual": residual,
            })
            print(f"{label:24} {status:>16} {rt:>9.2f} {str(obj):>5} "
                  f"{_b(cert):>9} {_b(acyc):>8} {roll:>11.3f} {res_v:>5} {res_e:>5}",
                  flush=True)

            # Loud per-instance failure detail (do not hide).
            if rep is not None and not (cert and acyc and roll == 1.0):
                print(f"    *** FAILURE {label}: cert={cert} acyc={acyc} rollout={roll}")
                for c in residual[:5]:
                    if isinstance(c, VertexConflict):
                        print(f"        V cell={c.vertex} agents=({c.agent_i},{c.agent_j}) "
                              f"ticks=({c.t_i},{c.t_j})")
                    else:
                        print(f"        E edge=({c.u}->{c.v}) agents=({c.agent_i},{c.agent_j}) "
                              f"ticks=({c.t_i},{c.t_j})")

        if n_optimal_this > 0:
            largest_optimal = max(largest_optimal, n)

    print("-" * 104)
    print(f"solved (plan realized)            : {n_solved}")
    verdict = "ALL PASS" if n_pass == n_solved else f"*** {n_solved - n_pass} FAILURE(S) ***"
    print(f"passed ALL of (3)(4)(5)           : {n_pass}/{n_solved}   {verdict}")
    print(f"largest n solving to OPTIMAL      : {largest_optimal}")
    n_to = sum(1 for r in rows if "timeout" in str(r["status"]))
    print(f"timeouts (>=0.5*budget)           : {n_to}")


def _b(x: object) -> str:
    if x is None:
        return "-"
    return "Y" if x else "NO"


if __name__ == "__main__":
    main()
