#!/usr/bin/env python3
"""Soundness-equivalence gate for wiring detect_conflicts_fast into the verifier.

plan_risk_upper_bound now calls detect_conflicts_fast (window-aware, exact)
instead of detect_conflicts. Since the conflict SET is identical, the risk value
must be IDENTICAL. This asserts, on nominal AND stn_certify-stretched plans:

  new_risk = plan_risk_upper_bound(plan, p_d, dw)              [fast detector]
  old_risk = sum(per_conflict_risk over detect_conflicts(plan, makespan))  [original]
  |new_risk - old_risk| < 1e-12  on every (plan, p_d, dw)

and that is_epsilon_certified gives identical accept/reject decisions vs the
old-detector reference. Also reports the end-to-end speedup of
stn_certify(probabilistic) at n=20/50 (the metric gating RQ3 runnability).

SANDBOX NOTE: MovingAI is blocked here -> SYNTHETIC congested instances (real
LaCAM* nominal plans). The risk-identity argument is detector-set equivalence
(already validated 240/240 exact on real data); this re-confirms it end-to-end.
The real-data verifier gate re-runs on the target machine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stn import _synthetic_instance  # noqa: E402

from slackcertify.certify.probabilistic import (  # noqa: E402
    is_epsilon_certified,
    per_conflict_risk,
    plan_risk_upper_bound,
)
from slackcertify.core.conflict import detect_conflicts  # noqa: E402
from slackcertify.core.plan import Plan  # noqa: E402
from slackcertify.repair.stn import STNInfeasible, stn_certify  # noqa: E402
from slackcertify.solvers.base import SolverError  # noqa: E402
from slackcertify.solvers.lacam import LaCAMStarSolver  # noqa: E402

N_AGENTS = [20, 50]
SEEDS = [1, 2, 3, 4, 5]
P_DS = [0.02, 0.05, 0.1]
DELTA_WINDOWS = [0, 1]
BOUNDED_STRETCH = [1, 2]
TOL = 1e-12
EPSILONS = [0.01, 0.05, 0.2]


def _old_risk(plan: Plan, p_d: float, dw: int) -> float:
    """Reference: original plan_risk_upper_bound (full detect_conflicts scan)."""
    horizon = max(plan.makespan, 0)
    pairs = detect_conflicts(plan, delta=horizon)
    return sum(per_conflict_risk(c, p_d, delta_window=dw) for c in pairs)


def main() -> None:
    solver = LaCAMStarSolver()
    max_diff = 0.0
    n_cmp = n_fail = n_decision_cmp = n_decision_fail = n_plans = 0

    print("=" * 96)
    print("RISK-EQUIVALENCE: plan_risk_upper_bound (fast detector) vs OLD (original detector)")
    print("SYNTHETIC instances (MovingAI blocked); real-data gate re-run on target. tol=1e-12")
    print("=" * 96)

    for n in N_AGENTS:
        for seed in SEEDS:
            graph, agents = _synthetic_instance(n, seed)
            try:
                nominal = solver.solve(graph, agents, time_limit_s=2.0)
            except SolverError as exc:
                print(f"[skip] n{n}/s{seed}: LaCAM* failed: {exc}")
                continue
            plans: list[tuple[str, Plan]] = [(f"n{n}/s{seed}/nominal", nominal)]
            for s in BOUNDED_STRETCH:
                try:
                    rep = stn_certify(nominal, delta=s, mode="bounded")
                    assert isinstance(rep, Plan)
                    plans.append((f"n{n}/s{seed}/bounded{s}", rep))
                except STNInfeasible:
                    pass

            for label, pl in plans:
                n_plans += 1
                for p_d in P_DS:
                    for dw in DELTA_WINDOWS:
                        new = plan_risk_upper_bound(pl, p_d, delta_window=dw)
                        old = _old_risk(pl, p_d, dw)
                        diff = abs(new - old)
                        max_diff = max(max_diff, diff)
                        n_cmp += 1
                        if diff >= TOL:
                            n_fail += 1
                            print(f"  !! RISK MISMATCH {label} p_d={p_d} dw={dw}: "
                                  f"new={new:.15g} old={old:.15g} diff={diff:.3e}")
                        # accept/reject decision identity
                        for eps in EPSILONS:
                            n_decision_cmp += 1
                            new_dec = is_epsilon_certified(pl, p_d, eps, delta_window=dw)
                            old_dec = old <= eps
                            if new_dec != old_dec:
                                n_decision_fail += 1
                                print(f"  !! DECISION MISMATCH {label} p_d={p_d} dw={dw} "
                                      f"eps={eps}: new={new_dec} old={old_dec}")
            print(f"  n{n}/s{seed}: {len(plans)} plans, running risk_fails={n_fail} "
                  f"decision_fails={n_decision_fail}", flush=True)

    print("-" * 96)
    print(f"plans tested              : {n_plans}")
    print(f"risk comparisons          : {n_cmp}")
    print(f"max |new - old| risk      : {max_diff:.3e}   (tol {TOL:g})")
    print(f"RISK MISMATCHES           : {n_fail}")
    print(f"decision comparisons      : {n_decision_cmp}")
    print(f"DECISION MISMATCHES       : {n_decision_fail}")
    print("RESULT                    : " + (
        "IDENTICAL risk + decisions" if (n_fail == 0 and n_decision_fail == 0)
        else "*** MISMATCH(ES) ***"))

    # End-to-end stn_certify(probabilistic) speedup gauge (now that the verifier
    # is fast). We report wall-time per solve; there is no slow baseline to
    # compare against here because the OLD verifier made these solves
    # intractable at n=50 -- that intractability was the motivation.
    print("\n" + "=" * 96)
    print("END-TO-END: stn_certify(probabilistic) wall-time with fast verifier")
    print("=" * 96)
    print(f"{'n':>4} {'p_d':>5} {'eps':>5} {'status':>12} {'s':>3} {'mkspn':>6} {'wall_s':>9}")
    for n in N_AGENTS:
        graph, agents = _synthetic_instance(n, 1)
        try:
            nominal = solver.solve(graph, agents, time_limit_s=3.0)
        except SolverError as exc:
            print(f"{n:>4}  LaCAM* failed: {exc}")
            continue
        for p_d, eps in [(0.05, 0.1), (0.05, 0.2)]:
            t0 = time.perf_counter()
            try:
                res = stn_certify(nominal, mode="probabilistic", p_d=p_d, epsilon=eps,
                                  return_diagnostics=True)
                assert isinstance(res, tuple)
                rep, diag = res
                wall = time.perf_counter() - t0
                print(f"{n:>4} {p_d:>5} {eps:>5} {'certified':>12} "
                      f"{str(diag['separation_s']):>3} {rep.makespan:>6} {wall:>9.2f}",
                      flush=True)
            except STNInfeasible:
                wall = time.perf_counter() - t0
                print(f"{n:>4} {p_d:>5} {eps:>5} {'infeasible':>12} {'-':>3} {'-':>6} "
                      f"{wall:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
