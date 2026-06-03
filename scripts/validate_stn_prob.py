#!/usr/bin/env python3
"""Probabilistic-mode validation harness for stn_certify (mirror of validate_stn.py).

The probabilistic mode binary-searches a global separation parameter ``s``,
using the REAL verifier ``plan_risk_upper_bound`` as the in-loop acceptance
test, so a returned plan cannot fail ``is_epsilon_certified``. For
representative ``(p_d, epsilon)`` pairs on warehouse-10-20-10-2-1 (seeds 1-10 x
n in {20,50}; synthetic congested fallback when the MovingAI host is blocked)
this confirms, per regime:

1. single-pass-per-candidate termination (the search itself always terminates);
2. ZERO soundness-gate failures: ``is_epsilon_certified`` is True on EVERY
   claimed certification (the required outcome);
3. acyclic repaired TPG;
4. the infeasibility rate (search exhausted: epsilon unachievable by wait-only
   repair within the feasible / capped separation range);
5. the median binary-search iteration count.

All verification is external; stn_certify is never self-trusted (its only
in-hot-path check IS the real verifier).
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Allow `python scripts/validate_stn_prob.py` to import the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stn import TIME_LIMIT_S, _load_instances  # noqa: E402

from slackcertify.certify.probabilistic import (  # noqa: E402
    is_epsilon_certified,
    plan_risk_upper_bound,
)
from slackcertify.core.plan import Plan  # noqa: E402
from slackcertify.core.tpg import build_tpg  # noqa: E402
from slackcertify.repair.stn import STNInfeasible, stn_certify  # noqa: E402
from slackcertify.solvers.base import SolverError  # noqa: E402
from slackcertify.solvers.lacam import LaCAMStarSolver  # noqa: E402

# Representative (p_d, epsilon) pairs: loose -> tight.
PAIRS = [(0.02, 0.3), (0.05, 0.1), (0.05, 0.2)]
DELTA_WINDOW = 0


def _eval_prob(plan: Plan, p_d: float, epsilon: float) -> dict[str, object]:
    """Run probabilistic stn_certify, then verify with the existing is_epsilon_certified."""
    t0 = time.perf_counter()
    infeasible = False
    reason = ""
    repaired: Plan | None = None
    diag: dict[str, object] = {}
    try:
        result = stn_certify(
            plan, DELTA_WINDOW, mode="probabilistic", p_d=p_d, epsilon=epsilon,
            return_diagnostics=True,
        )
        assert isinstance(result, tuple)
        repaired, diag = result
    except STNInfeasible as exc:
        infeasible = True
        reason = exc.reason
    wall = time.perf_counter() - t0

    gate: bool | None = None
    acyclic: bool | None = None
    risk_ub = 0.0
    rep_makespan = 0
    if repaired is not None:
        rep_makespan = repaired.makespan
        gate = is_epsilon_certified(repaired, p_d, epsilon, delta_window=DELTA_WINDOW)
        acyclic = build_tpg(repaired).is_acyclic()
        risk_ub = plan_risk_upper_bound(repaired, p_d, delta_window=DELTA_WINDOW)

    certified = (not infeasible) and gate is True and acyclic is True
    return {
        "infeasible": infeasible,
        "reason": reason,
        "wall": wall,
        "separation_s": diag.get("separation_s"),
        "total_waits": diag.get("total_waits", 0),
        "search_iters": diag.get("search_iters"),
        "solver": diag.get("solver", "-"),
        "gate": gate,
        "acyclic": acyclic,
        "risk_ub": risk_ub,
        "rep_makespan": rep_makespan,
        "certified": certified,
    }


def main() -> None:
    instances, synthetic = _load_instances()
    solver = LaCAMStarSolver()

    if synthetic:
        print("=" * 110)
        print("!! WARNING: MovingAI host blocked (benchmarks/maps empty). Falling back to")
        print("!! SELF-GENERATED congested instances solved by the real LaCAM* binary.")
        print("!! Probabilistic STN / verifier logic identical; only the instance source differs.")
        print("=" * 110)

    # Solve each instance once; reuse the nominal plan across all pairs.
    solved: list[tuple[str, Plan]] = []
    for label, _n, _seed, graph, agents in instances:
        try:
            plan = solver.solve(graph, agents, time_limit_s=TIME_LIMIT_S)
        except SolverError as exc:
            print(f"[skip] {label}: LaCAM* failed: {exc}")
            continue
        solved.append((label, plan))

    for p_d, epsilon in PAIRS:
        rows: list[dict[str, object]] = []
        for label, plan in solved:
            rows.append({"label": label, **_eval_prob(plan, p_d, epsilon)})

        print("\n" + "=" * 118)
        print(f"PROBABILISTIC  p_d={p_d}  epsilon={epsilon}  (delta_window={DELTA_WINDOW})")
        print("-" * 118)
        print(
            f"{'instance':30} {'infeas':>6} {'s':>3} {'iters':>5} {'gate':>5} {'acyc':>5} "
            f"{'riskUB':>9} {'eps':>6} {'waits':>6} {'mkspn':>6} {'wall_s':>8}"
        )
        print("-" * 118)
        for r in rows:
            gate = "-" if r["gate"] is None else ("yes" if r["gate"] else "NO")
            acyc = "-" if r["acyclic"] is None else ("yes" if r["acyclic"] else "NO")
            s_disp = "-" if r["separation_s"] is None else str(r["separation_s"])
            it_disp = "-" if r["search_iters"] is None else str(r["search_iters"])
            print(
                f"{r['label']:30} {('YES' if r['infeasible'] else 'no'):>6} {s_disp:>3} "
                f"{it_disp:>5} {gate:>5} {acyc:>5} {float(r['risk_ub']):>9.5f} "
                f"{epsilon:>6.3f} {r['total_waits']:>6} {r['rep_makespan']:>6} "
                f"{float(r['wall']):>8.2f}"
            )

        n_total = len(rows)
        certified = sum(1 for r in rows if r["certified"])
        infeasible = sum(1 for r in rows if r["infeasible"])
        iters = [int(r["search_iters"]) for r in rows if r["search_iters"] is not None]
        med_iters = statistics.median(iters) if iters else float("nan")
        bugs = [
            r for r in rows
            if (not r["infeasible"]) and (r["gate"] is not True or r["acyclic"] is not True)
        ]

        print("-" * 118)
        print(f"  N = {n_total}")
        print(f"  STN certified (gate True AND acyclic)   : {certified}/{n_total} "
              f"({certified / n_total:.1%})" if n_total else "")
        print(f"  STN infeasible (s-search exhausted)     : {infeasible}/{n_total}  "
              f"-- INFEASIBILITY RATE = {infeasible / n_total:.1%}" if n_total else "")
        print(f"  SOUNDNESS-GATE FAILURES (must be 0)      : {len(bugs)}")
        print(f"  median binary-search iterations         : {med_iters}")

        if bugs:
            print("\n" + "#" * 118)
            print(f"#### SOUNDNESS-GATE FAILURE: {len(bugs)} instance(s) where STN returned a plan "
                  "but is_epsilon_certified is False (or TPG cyclic):")
            for r in bugs:
                print(f"####   {r['label']}: gate={r['gate']} acyclic={r['acyclic']} "
                      f"riskUB={float(r['risk_ub']):.5f} > eps={epsilon} s={r['separation_s']}")
            print("#" * 118)
        else:
            print("\n  [OK] ZERO soundness-gate failures: every returned plan is "
                  "epsilon-certified and acyclic.")


if __name__ == "__main__":
    main()
