#!/usr/bin/env python3
"""Equivalence + speedup gate for detect_conflicts_fast vs detect_conflicts.

detect_conflicts is the most load-bearing function in the codebase (RQ1 residual
checks, the probabilistic verifier, the ILP gate, the Plan validator). The
windowed detect_conflicts_fast must reproduce its output EXACTLY. This harness:

  (A) EQUIVALENCE: for nominal AND stn_certify-stretched plans, over
      delta in {0, 1, 2, makespan}, asserts
        set(fast) == set(old) AND len(fast) == len(old) AND list(fast)==list(old)
      (list order checked too, so even order-sensitive callers are safe).
      On mismatch it prints the plan/delta and the symmetric-difference
      conflicts (added/dropped) -- the test is NOT adjusted.
  (B) RQ1-NOT-BROKEN: on stn_certify bounded plans (delta=2), confirms
      detect_conflicts_fast residual == 0 exactly where detect_conflicts gives
      0 (and never reports a residual where the old said 0, or vice versa).
  (C) SPEEDUP: at delta=makespan (the probabilistic verifier's regime), reports
      old vs fast wall-time at n=20/50/100.

SANDBOX NOTE: MovingAI is blocked here, so this falls back to SYNTHETIC
congested instances (real LaCAM* nominal plans). The equivalence is therefore
validated on SYNTHETIC data; the REAL-data gate (warehouse-10-20-10-2-1,
random-32-32-20) must be re-run on the target machine. No real-data validation
is claimed here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stn import _synthetic_instance  # noqa: E402

from slackcertify.core.conflict import detect_conflicts, detect_conflicts_fast  # noqa: E402
from slackcertify.core.plan import Plan  # noqa: E402
from slackcertify.repair.stn import STNInfeasible, stn_certify  # noqa: E402
from slackcertify.solvers.base import SolverError  # noqa: E402
from slackcertify.solvers.lacam import LaCAMStarSolver  # noqa: E402

N_EQUIV = [20, 50, 100]
SEEDS = [1, 2, 3, 4, 5]
BOUNDED_STRETCH = [1, 2, 5]
# Probabilistic-stretched plans are intentionally OMITTED from the equivalence
# sweep: building them calls the slow plan_risk_upper_bound (delta=makespan
# detect_conflicts) inside a binary search and dominates runtime at n>=50 (the
# very bottleneck this task removes). They add no detector coverage beyond what
# the nominal + bounded{1,2,5} plans already exercise (varied makespan,
# goal-rest, convention-B). A single probabilistic plan is checked separately
# below to confirm the detector also matches on a probabilistic-shaped plan.
PROB_STRETCH: list[tuple[float, float]] = []


def _check(plan: Plan, delta: int, label: str, stats: dict) -> None:
    old = detect_conflicts(plan, delta)
    fast = detect_conflicts_fast(plan, delta)
    so, sf = set(old), set(fast)
    stats["compared"] += 1
    set_eq = so == sf
    len_eq = len(old) == len(fast)
    list_eq = old == fast
    if not (set_eq and len_eq and list_eq):
        stats["failures"] += 1
        print(f"  !! MISMATCH {label} delta={delta}: old={len(old)} fast={len(fast)} "
              f"set_eq={set_eq} list_eq={list_eq}")
        for c in list(so - sf)[:5]:
            print(f"      only-old : {c}")
        for c in list(sf - so)[:5]:
            print(f"      only-fast: {c}")
    else:
        stats["max_conflicts"] = max(stats["max_conflicts"], len(old))


def main() -> None:
    solver = LaCAMStarSolver()
    stats = {"compared": 0, "failures": 0, "max_conflicts": 0, "plans": 0, "convb": 0}
    rq1_checked = rq1_agree = 0

    print("=" * 96)
    print("EQUIVALENCE: detect_conflicts_fast vs detect_conflicts (set + len + list order)")
    print("SYNTHETIC congested instances (MovingAI blocked); real-data gate re-run on target.")
    print("=" * 96)

    def has_revisit(pl: Plan) -> bool:
        for p in pl.paths:
            v = p.vertices
            g = v[-1]
            i = len(v) - 1
            while i > 0 and v[i - 1] == g:
                i -= 1
            if any(c == g for c in v[:i]):
                return True
        return False

    for n in N_EQUIV:
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
            for p_d, eps in PROB_STRETCH:
                try:
                    rep = stn_certify(nominal, mode="probabilistic", p_d=p_d, epsilon=eps)
                    assert isinstance(rep, Plan)
                    plans.append((f"n{n}/s{seed}/prob{p_d}_{eps}", rep))
                except STNInfeasible:
                    pass

            for label, pl in plans:
                stats["plans"] += 1
                if has_revisit(pl):
                    stats["convb"] += 1
                for delta in (0, 1, 2, pl.makespan):
                    _check(pl, delta, label, stats)

                # (B) RQ1-not-broken: bounded delta=2 residual agreement.
                if "/bounded2" in label:
                    rq1_checked += 1
                    old0 = len(detect_conflicts(pl, 2)) == 0
                    fast0 = len(detect_conflicts_fast(pl, 2)) == 0
                    if old0 == fast0:
                        rq1_agree += 1
                    else:
                        print(f"  !! RQ1 DISAGREE {label}: "
                              f"old_residual0={old0} fast_residual0={fast0}")
            print(f"  n{n}/s{seed}: {len(plans)} plans checked, "
                  f"running failures={stats['failures']}", flush=True)

    # One probabilistic-shaped plan at small n (cheap) so a binary-search-
    # stretched plan is also covered without the n>=50 verifier-cost hang.
    try:
        g20, ag20 = _synthetic_instance(20, 1)
        nom20 = solver.solve(g20, ag20, time_limit_s=2.0)
        rep = stn_certify(nom20, mode="probabilistic", p_d=0.05, epsilon=0.2)
        assert isinstance(rep, Plan)
        stats["plans"] += 1
        if has_revisit(rep):
            stats["convb"] += 1
        for delta in (0, 1, 2, rep.makespan):
            _check(rep, delta, "n20/s1/prob0.05_0.2", stats)
        print(f"  probabilistic n20/s1: checked, running failures={stats['failures']}",
              flush=True)
    except (SolverError, STNInfeasible) as exc:
        print(f"  probabilistic n20/s1: skipped ({type(exc).__name__})")

    print("-" * 96)
    print(f"plans tested              : {stats['plans']}")
    print(f"convention-B plans        : {stats['convb']}/{stats['plans']}")
    print(f"(plan,delta) comparisons  : {stats['compared']}")
    print(f"max conflicts in a compare: {stats['max_conflicts']}")
    print(f"MISMATCHES                : {stats['failures']}")
    print("EQUIVALENCE               : " + (
        "ALL EXACTLY EQUAL (set + len + list order)" if stats["failures"] == 0
        else f"*** {stats['failures']} MISMATCH(ES) ***"))
    print(f"RQ1-not-broken (bnd d=2)  : {rq1_agree}/{rq1_checked} residual-zero agreement"
          + ("  OK" if rq1_agree == rq1_checked else "  *** BROKEN ***"))

    # (C) Speedup at delta=makespan.
    print("\n" + "=" * 96)
    print("SPEEDUP at delta=makespan (probabilistic-verifier regime): old vs fast")
    print("=" * 96)
    print(f"{'n':>4} {'mkspn':>6} {'#confl':>8} {'old_s':>10} {'fast_s':>10} "
          f"{'speedup':>9} {'match':>6}")
    for n in N_EQUIV:
        graph, agents = _synthetic_instance(n, 1)
        try:
            nominal = solver.solve(graph, agents, time_limit_s=3.0)
        except SolverError as exc:
            print(f"{n:>4}  LaCAM* failed: {exc}")
            continue
        mk = nominal.makespan
        t0 = time.perf_counter()
        old = detect_conflicts(nominal, mk)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        fast = detect_conflicts_fast(nominal, mk)
        t_fast = time.perf_counter() - t0
        spd = t_old / t_fast if t_fast > 0 else float("inf")
        match = set(old) == set(fast) and old == fast
        print(f"{n:>4} {mk:>6} {len(old):>8} {t_old:>10.3f} {t_fast:>10.4f} "
              f"{spd:>8.1f}x {str(match):>6}", flush=True)


if __name__ == "__main__":
    main()
