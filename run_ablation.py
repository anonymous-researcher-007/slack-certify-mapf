#!/usr/bin/env python3
"""Full ablation sweep (decide reportable subset from results).

A2 (front-end solver): EECBS vs LaCAM* vs PIBT, bounded Delta-certificate.
A3 (certificate mode): bounded (Delta=2) vs probabilistic (epsilon) on LaCAM*.
For each cell: solve nominal -> certify -> record success (certified+executable
via rollout), SoC overhead vs that solver's nominal, infeasibility.
Two maps (consistent with RQ1-RQ4), n in {100, 200}, 25 seeds.
"""
import csv
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path as P

MAPS = ["warehouse-10-20-10-2-1", "random-32-32-20"]
NS = [100, 200]
SEEDS = list(range(1, 26))
DELTA = 2
EPS = 0.1            # for A3 epsilon-mode
P_D = 0.03
K_ROLLOUTS = 500
CELL_TIMEOUT_S = 180
N_WORKERS = 16
OUT = "results/ablation.csv"

# variants: (axis, label, solver_name, mode)
VARIANTS = [
    ("A2", "eecbs",     "eecbs",      "bounded"),
    ("A2", "lacam_star","lacam_star", "bounded"),
    ("A3", "delta",     "lacam_star", "bounded"),       # same as A2 lacam_star, kept for table clarity
    ("A3", "epsilon",   "lacam_star", "probabilistic"),
]

class _Timeout(Exception): pass
def _alarm(s, f): raise _Timeout()

def _cell(args):
    import numpy as np
    axis, label, solver_name, mode, mp, n, seed = args
    from slackcertify.benchmarks.scenarios import ScenarioRegistry
    from slackcertify.delay.bernoulli import BernoulliDelayModel
    from slackcertify.delay.bounded import BoundedDelayModel
    from slackcertify.repair.stn import STNInfeasible, stn_certify
    from slackcertify.simulate.rollout import monte_carlo_rollout
    from slackcertify.solvers import EECBSSolver, LaCAMStarSolver, PIBTSolver
    SOLVERS = {"eecbs": EECBSSolver, "lacam_star": LaCAMStarSolver, "pibt": PIBTSolver}
    rec = {"axis": axis, "variant": label, "solver": solver_name, "mode": mode,
           "map": mp, "n": n, "seed": seed, "status": "", "success": "",
           "nominal_soc": "", "soc_overhead_pct": "", "sep_s": "", "time_s": ""}
    try:
        reg = ScenarioRegistry(root=P("benchmarks"))
        g, a = reg.load_instance(mp, n, seed)
        _budget = 2.0 if solver_name == "lacam_star" else 30.0
        plan = SOLVERS[solver_name]().solve(g, a, time_limit_s=_budget)
    except Exception as e:
        rec["status"] = f"solve_err:{type(e).__name__}"; return rec
    nominal_soc = plan.sum_of_costs if hasattr(plan, "sum_of_costs") else None
    rec["nominal_soc"] = nominal_soc
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time(); signal.alarm(CELL_TIMEOUT_S)
    try:
        if mode == "bounded":
            rep = stn_certify(plan, DELTA)
            dm = BoundedDelayModel(delta=DELTA)
        else:
            rep, d = stn_certify(plan, mode="probabilistic", p_d=P_D, epsilon=EPS,
                                 return_diagnostics=True)
            rec["sep_s"] = d["separation_s"]
            dm = BernoulliDelayModel(p_d=P_D)
        signal.alarm(0)
        rep_soc = rep.sum_of_costs if hasattr(rep, "sum_of_costs") else None
        if nominal_soc and rep_soc:
            rec["soc_overhead_pct"] = f"{100*(rep_soc-nominal_soc)/nominal_soc:.2f}"
        roll = monte_carlo_rollout(rep, dm, K=K_ROLLOUTS, n_jobs=1,
                                   rng=np.random.default_rng(seed))
        rec["success"] = f"{roll.success_rate:.4f}"
        rec["status"] = "ok"; rec["time_s"] = f"{time.time()-t0:.2f}"
    except STNInfeasible:
        signal.alarm(0); rec["status"] = "infeasible"; rec["time_s"] = f"{time.time()-t0:.2f}"
    except _Timeout:
        rec["status"] = "timeout"; rec["time_s"] = f">{CELL_TIMEOUT_S}"
    except Exception as e:
        signal.alarm(0); rec["status"] = f"err:{type(e).__name__}"
    return rec

def main():
    cells = [(ax, lb, sv, md, mp, n, s)
             for (ax, lb, sv, md) in VARIANTS
             for mp in MAPS for n in NS for s in SEEDS]
    print(f"{len(cells)} cells on {N_WORKERS} workers", flush=True)
    P("results").mkdir(exist_ok=True)
    rows, done, t0 = [], 0, time.time()
    fields = ["axis","variant","solver","mode","map","n","seed","status",
              "success","nominal_soc","soc_overhead_pct","sep_s","time_s"]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_cell, c): c for c in cells}
        for fut in as_completed(futs):
            rec = fut.result(); rows.append(rec); done += 1
            print(f"[{done}/{len(cells)}] {rec['axis']} {rec['variant']:10} "
                  f"{rec['map'][:10]} n={rec['n']:<3} s={rec['seed']:<2}: "
                  f"{rec['status']:10} succ={rec['success']} soc%={rec['soc_overhead_pct']}", flush=True)
            if done % 20 == 0 or done == len(cells):
                with open(OUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min, {len(rows)} cells -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
