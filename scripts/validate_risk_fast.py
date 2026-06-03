#!/usr/bin/env python3
"""Equivalence + speedup harness for plan_risk_upper_bound_fast (the soundness gate).

Confirms plan_risk_upper_bound_fast == plan_risk_upper_bound (within 1e-9) on
real-shaped LaCAM* plans stretched by stn_certify, across (p_d, delta_window)
including p_d=0.1 (slowest decay => largest W). The slow original is ground
truth. Also reports the chosen window W per (p_d, dw, makespan) and the
old-vs-new wall-time speedup at n in {20, 50, 200}.

stn_certify may legitimately raise STNInfeasible (rotational deadlock); such
plans are skipped (they cannot be built, so there is nothing to compare).

MovingAI host is blocked here, so instances use the synthetic congested
warehouse fallback (real LaCAM* nominal plans); logic is identical to the
real-warehouse path.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stn import _synthetic_instance  # noqa: E402

from slackcertify.certify.probabilistic import (  # noqa: E402
    _risk_window,
    per_conflict_risk,
    plan_risk_upper_bound,
    plan_risk_upper_bound_fast,
)
from slackcertify.core.conflict import detect_conflicts  # noqa: E402
from slackcertify.core.plan import Plan  # noqa: E402
from slackcertify.repair.stn import STNInfeasible, stn_certify  # noqa: E402
from slackcertify.solvers.base import SolverError  # noqa: E402
from slackcertify.solvers.lacam import LaCAMStarSolver  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
N_EQUIV = [20, 50]
SEPARATIONS = [1, 2, 3, 5, 8]
P_DS = [0.02, 0.05, 0.1]
DELTA_WINDOWS = [0, 1]
TOL = 1e-9
SPEEDUP_CAP_S = 900.0
OUT = open("/tmp/risk_fast_report.txt", "w")  # noqa: SIM115


def log(*a: object) -> None:
    print(*a)
    print(*a, file=OUT, flush=True)


def _has_goal_revisit(plan: Plan) -> bool:
    """True if some agent leaves and re-enters its goal (convention B)."""
    for p in plan.paths:
        v = p.vertices
        goal = v[-1]
        i = len(v) - 1
        while i > 0 and v[i - 1] == goal:
            i -= 1
        if any(c == goal for c in v[:i]):
            return True
    return False


class _TimeoutError(Exception):
    pass


def _time_capped(fn, cap_s: float):  # noqa: ANN001, ANN202
    """Run fn(), returning (seconds, value) or (None, None) on timeout."""
    def _h(signum: int, frame: object) -> None:
        raise _TimeoutError

    prev = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, cap_s)
    t0 = time.perf_counter()
    try:
        val = fn()
        return time.perf_counter() - t0, val
    except _TimeoutError:
        return None, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _speedup_table(solver: LaCAMStarSolver) -> None:
    log("\n" + "=" * 100)
    log("SPEEDUP  (p_d=0.05, dw=1, stretched at s=3): old vs fast wall-time")
    log("=" * 100)
    log(f"{'n':>4} {'mkspn':>6} {'old_s':>11} {'fast_s':>10} {'speedup':>9} {'W':>4} {'match':>6}")
    for n in [20, 50, 200]:
        graph, agents = _synthetic_instance(n, 1)
        try:
            nominal = solver.solve(graph, agents, time_limit_s=3.0)
        except SolverError as exc:
            log(f"{n:>4}  LaCAM* failed: {exc}")
            continue
        try:
            rep = stn_certify(nominal, delta=3, mode="bounded")
        except STNInfeasible as exc:
            log(f"{n:>4}  stn_certify infeasible: {exc}")
            continue
        assert isinstance(rep, Plan)
        w = _risk_window(0.05, 1, max(rep.makespan, 0))
        cap = SPEEDUP_CAP_S
        t_fast, fast = _time_capped(lambda r=rep: plan_risk_upper_bound_fast(r, 0.05, 1), cap)
        t_old, old = _time_capped(lambda r=rep: plan_risk_upper_bound(r, 0.05, 1), cap)
        if t_old is None:
            log(f"{n:>4} {rep.makespan:>6} {'>900(TMO)':>11} {t_fast:>10.4f} "
                f"{'>'+str(int(SPEEDUP_CAP_S/max(t_fast,1e-9)))+'x':>9} {w:>4} {'n/a':>6}")
        else:
            spd = t_old / t_fast if t_fast and t_fast > 0 else float("inf")
            match = abs(old - fast) < TOL
            log(f"{n:>4} {rep.makespan:>6} {t_old:>11.3f} {t_fast:>10.4f} "
                f"{spd:>8.1f}x {w:>4} {str(match):>6}")


def _equivalence(solver: LaCAMStarSolver) -> tuple[float, int, int, int, int, dict]:
    max_abs_diff = 0.0
    n_compared = n_failures = n_conv_b = n_plans = 0
    windows: dict[tuple[float, int, int], int] = {}

    log("=" * 100)
    log("EQUIVALENCE TEST  (plan_risk_upper_bound_fast vs plan_risk_upper_bound, tol=1e-9)")
    log("synthetic congested warehouse (MovingAI host blocked); real LaCAM* nominal plans")
    log("=" * 100)

    for n in N_EQUIV:
        for seed in SEEDS:
            graph, agents = _synthetic_instance(n, seed)
            try:
                nominal = solver.solve(graph, agents, time_limit_s=2.0)
            except SolverError as exc:
                log(f"[skip] n={n} seed={seed}: LaCAM* failed: {exc}")
                continue
            for s in SEPARATIONS:
                try:
                    rep = stn_certify(nominal, delta=s, mode="bounded")
                except STNInfeasible as exc:
                    log(f"  [skip] n={n} seed={seed} s={s}: STNInfeasible ({exc})")
                    continue
                assert isinstance(rep, Plan)
                n_plans += 1
                conv_b = _has_goal_revisit(rep)
                n_conv_b += int(conv_b)
                pairs_full = detect_conflicts(rep, delta=max(rep.makespan, 0))
                for p_d in P_DS:
                    for dw in DELTA_WINDOWS:
                        old = sum(per_conflict_risk(c, p_d, delta_window=dw) for c in pairs_full)
                        fast = plan_risk_upper_bound_fast(rep, p_d, dw)
                        diff = abs(old - fast)
                        max_abs_diff = max(max_abs_diff, diff)
                        n_compared += 1
                        windows[(p_d, dw, rep.makespan)] = _risk_window(
                            p_d, dw, max(rep.makespan, 0)
                        )
                        if diff >= TOL:
                            n_failures += 1
                            log(f"  !! MISMATCH n={n} seed={seed} s={s} p_d={p_d} dw={dw} "
                                f"mkspn={rep.makespan} old={old:.12g} fast={fast:.12g} "
                                f"diff={diff:.3e}")
                log(f"  n={n:>3} seed={seed} s={s}: mkspn={rep.makespan:>4} "
                    f"convB={'Y' if conv_b else 'n'} #pairs_full={len(pairs_full):>6} "
                    f"max_diff_so_far={max_abs_diff:.2e}")

    return max_abs_diff, n_compared, n_failures, n_conv_b, n_plans, windows


def main() -> None:
    solver = LaCAMStarSolver()

    # Speedup first (the key missing deliverable; n=200 old call may TMO).
    _speedup_table(solver)

    max_abs_diff, n_compared, n_failures, n_conv_b, n_plans, windows = _equivalence(solver)

    log("-" * 100)
    log(f"plans tested                : {n_plans}")
    log(f"convention-B plans (revisit): {n_conv_b}/{n_plans}")
    log(f"(p_d,dw) comparisons        : {n_compared}")
    log(f"max |fast - old|            : {max_abs_diff:.3e}  (tolerance {TOL:g})")
    log(f"MISMATCHES (>= tol)         : {n_failures}")
    log("RESULT                      : " + ("ALL MATCH within 1e-9" if n_failures == 0
                                            else f"*** {n_failures} MISMATCH(ES) ***"))

    log("\n" + "=" * 100)
    log("WINDOW W per (p_d, delta_window)  [min..max over makespans tested]")
    log("=" * 100)
    by_pddw: dict[tuple[float, int], list[tuple[int, int]]] = {}
    for (p_d, dw, mk), w in windows.items():
        by_pddw.setdefault((p_d, dw), []).append((mk, w))
    for key in sorted(by_pddw):
        p_d, dw = key
        ws = [w for _, w in by_pddw[key]]
        mks = [mk for mk, _ in by_pddw[key]]
        log(f"  p_d={p_d}  dw={dw}:  W in [{min(ws)}..{max(ws)}]  "
            f"(makespans {min(mks)}..{max(mks)})")

    OUT.close()


if __name__ == "__main__":
    main()
