#!/usr/bin/env python3
import csv
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path as P

GRID = {
    "warehouse-10-20-10-2-1": [20, 50, 100],
    "random-32-32-20":        [20, 50],
}
EPSILONS = [0.05, 0.1, 0.2, 0.3]
SEEDS = list(range(1, 26))
P_D = 0.03
K_ROLLOUTS = 500
CELL_TIMEOUT_S = 180
N_WORKERS = 16
OUT = "results/rq3_tightness.csv"

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout()

def _cell(args):
    mp, n, seed, eps = args
    from slackcertify.benchmarks.scenarios import ScenarioRegistry
    from slackcertify.delay.bernoulli import BernoulliDelayModel
    from slackcertify.repair.stn import STNInfeasible, stn_certify
    from slackcertify.simulate.rollout import monte_carlo_rollout
    from slackcertify.solvers.lacam import LaCAMStarSolver
    rec = {"map": mp, "n": n, "seed": seed, "target_eps": eps, "p_d": P_D,
           "status": "", "risk_ub": "", "sep_s": "", "empirical_collision": "",
           "cert_time_s": ""}
    try:
        reg = ScenarioRegistry(root=P("benchmarks"))
        g, a = reg.load_instance(mp, n, seed)
        plan = LaCAMStarSolver().solve(g, a, time_limit_s=2.0)
    except Exception as e:
        rec["status"] = f"setup_err:{type(e).__name__}"
        return rec
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time()
    signal.alarm(CELL_TIMEOUT_S)
    try:
        rep, d = stn_certify(plan, mode="probabilistic", p_d=P_D,
                             epsilon=eps, return_diagnostics=True)
        signal.alarm(0)
        rec["cert_time_s"] = f"{time.time()-t0:.2f}"
        rec["risk_ub"] = f"{d['risk_ub']:.6f}"
        rec["sep_s"] = d["separation_s"]
        roll = monte_carlo_rollout(rep, BernoulliDelayModel(p_d=P_D),
                                   K=K_ROLLOUTS, n_jobs=1)
        rec["empirical_collision"] = f"{1.0 - roll.success_rate:.6f}"
        rec["status"] = "ok"
    except STNInfeasible:
        signal.alarm(0); rec["status"] = "infeasible"
        rec["cert_time_s"] = f"{time.time()-t0:.2f}"
    except _Timeout:
        rec["status"] = "timeout"; rec["cert_time_s"] = f">{CELL_TIMEOUT_S}"
    except Exception as e:
        signal.alarm(0); rec["status"] = f"err:{type(e).__name__}"
    return rec

def main():
    cells = [(mp, n, seed, eps)
             for mp, ns in GRID.items() for n in ns
             for seed in SEEDS for eps in EPSILONS]
    print(f"{len(cells)} cells on {N_WORKERS} workers", flush=True)
    P("results").mkdir(exist_ok=True)
    rows, done, t0 = [], 0, time.time()
    fields = ["map","n","seed","target_eps","p_d","status","risk_ub","sep_s",
              "empirical_collision","cert_time_s"]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_cell, c): c for c in cells}
        for fut in as_completed(futs):
            rec = fut.result(); rows.append(rec); done += 1
            print(f"[{done}/{len(cells)}] {rec['map'][:10]} n={rec['n']:<3} "
                  f"s={rec['seed']:<2} eps={rec['target_eps']}: {rec['status']:10} "
                  f"risk={rec['risk_ub']} emp={rec['empirical_collision']} "
                  f"({rec['cert_time_s']}s)", flush=True)
            if done % 20 == 0 or done == len(cells):
                with open(OUT, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader(); w.writerows(rows)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min, {len(rows)} cells -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
