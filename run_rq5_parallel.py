#!/usr/bin/env python3
"""RQ5 persistent-delay stress: certify bounded, calibrate matched p_d, roll out P2 vs P3."""
import csv, time, signal, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path as P

GRID = {"warehouse-10-20-10-2-1": [20, 50, 100], "random-32-32-20": [20, 50]}
P_TRIGGERS = [0.002, 0.005, 0.01, 0.02]
MIN_LEN, MAX_LEN = 2, 4
DELTA = 2
SEEDS = list(range(1, 26))
K_ROLLOUTS = 500
CELL_TIMEOUT_S = 120
N_WORKERS = 16
OUT = "results/rq5_persistent.csv"

class _Timeout(Exception): pass
def _alarm(s, f): raise _Timeout()

def _cell(args):
    import numpy as np
    mp, n, seed, ptrig = args
    from slackcertify.benchmarks.scenarios import ScenarioRegistry
    from slackcertify.solvers.lacam import LaCAMStarSolver
    from slackcertify.repair.stn import stn_certify, STNInfeasible
    from slackcertify.simulate.rollout import monte_carlo_rollout
    from slackcertify.delay.bernoulli import BernoulliDelayModel
    from slackcertify.delay.persistent import PersistentDelayModel
    from slackcertify.delay.calibration import calibrate_bernoulli_to_persistent
    rec = {"map": mp, "n": n, "seed": seed, "p_trigger": ptrig,
           "min_len": MIN_LEN, "max_len": MAX_LEN, "p_d_calib": "",
           "status": "", "success_P2": "", "success_P3": "", "time_s": ""}
    try:
        reg = ScenarioRegistry(root=P("benchmarks"))
        g, a = reg.load_instance(mp, n, seed)
        plan = LaCAMStarSolver().solve(g, a, time_limit_s=2.0)
    except Exception as e:
        rec["status"] = f"setup_err:{type(e).__name__}"; return rec
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time(); signal.alarm(CELL_TIMEOUT_S)
    try:
        rep = stn_certify(plan, DELTA)
        signal.alarm(0)
        pm = PersistentDelayModel(p_trigger=ptrig, min_len=MIN_LEN, max_len=MAX_LEN)
        rng = np.random.default_rng(12345 + seed)
        p_d = calibrate_bernoulli_to_persistent(pm, plan_length=rep.makespan, rng=rng)
        rec["p_d_calib"] = f"{p_d:.5f}"
        r2 = monte_carlo_rollout(rep, BernoulliDelayModel(p_d=p_d), K=K_ROLLOUTS, n_jobs=1,
                                 rng=np.random.default_rng(7 + seed))
        r3 = monte_carlo_rollout(rep, pm, K=K_ROLLOUTS, n_jobs=1,
                                 rng=np.random.default_rng(8 + seed))
        rec["success_P2"] = f"{r2.success_rate:.4f}"
        rec["success_P3"] = f"{r3.success_rate:.4f}"
        rec["status"] = "ok"; rec["time_s"] = f"{time.time()-t0:.2f}"
    except STNInfeasible:
        signal.alarm(0); rec["status"] = "infeasible"; rec["time_s"] = f"{time.time()-t0:.2f}"
    except _Timeout:
        rec["status"] = "timeout"; rec["time_s"] = f">{CELL_TIMEOUT_S}"
    except Exception as e:
        signal.alarm(0); rec["status"] = f"err:{type(e).__name__}"
    return rec

def main():
    cells = [(mp, n, seed, pt) for mp, ns in GRID.items() for n in ns
             for seed in SEEDS for pt in P_TRIGGERS]
    print(f"{len(cells)} cells on {N_WORKERS} workers", flush=True)
    P("results").mkdir(exist_ok=True)
    rows, done, t0 = [], 0, time.time()
    fields = ["map","n","seed","p_trigger","min_len","max_len","p_d_calib",
              "status","success_P2","success_P3","time_s"]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_cell, c): c for c in cells}
        for fut in as_completed(futs):
            rec = fut.result(); rows.append(rec); done += 1
            print(f"[{done}/{len(cells)}] {rec['map'][:10]} n={rec['n']:<3} "
                  f"s={rec['seed']:<2} pt={rec['p_trigger']}: {rec['status']:10} "
                  f"P2={rec['success_P2']} P3={rec['success_P3']} pd={rec['p_d_calib']}", flush=True)
            if done % 20 == 0 or done == len(cells):
                with open(OUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min, {len(rows)} cells -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
