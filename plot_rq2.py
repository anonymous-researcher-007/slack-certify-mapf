"""Fig. 2 - RQ2: optimality gap vs. agent count (left) and runtime STN vs ILP (right).
Reads results/rq2_ilp.csv. Saves figures/fig2_rq2_gap_runtime.png at 300 dpi.
No external LaTeX required.
"""
import collections
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/rq2_ilp.csv"
OUT = "figures/fig2_rq2_gap_runtime.png"
DPI = 300
STN_C, ILP_C = "#1F6FB2", "#C0392B"

rows = list(csv.DictReader(open(CSV)))

# pair by (n, seed) via total_waits; gather runtimes
sc_w, il_w = {}, {}
stn_rt, ilp_rt = collections.defaultdict(list), collections.defaultdict(list)
for r in rows:
    if r["status"] != "ok":
        continue
    n = int(r["n_agents"]); key = (n, r["scen_seed"])
    tw = r.get("total_waits")
    if r["method"] == "slack_certify_bounded":
        if tw not in (None, ""): sc_w[key] = float(tw)
        if r.get("certifier_runtime_s"): stn_rt[n].append(float(r["certifier_runtime_s"]) * 1000)
    elif r["method"] == "ilp_exact":
        if tw not in (None, ""): il_w[key] = float(tw)
        try:
            d = json.loads(r.get("diagnostics", "{}") or "{}"); rt = d.get("ilp_runtime_s")
            if rt is not None: ilp_rt[n].append(float(rt) * 1000)
        except Exception:
            pass

gaps_by_n = collections.defaultdict(list)
for key, s in sc_w.items():
    if key not in il_w: continue
    i = il_w[key]
    g = (s - i) / i * 100 if i > 0 else (0.0 if s == 0 else np.nan)
    if not np.isnan(g): gaps_by_n[int(key[0])].append(g)

ns = sorted(gaps_by_n)

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.linewidth": 0.9,
})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8))

# LEFT: gap distribution per n (strip + mean marker)
for xi, n in enumerate(ns):
    vals = gaps_by_n[n]
    jitter = (np.random.RandomState(n).rand(len(vals)) - 0.5) * 0.28
    axL.scatter(np.full(len(vals), xi) + jitter, vals, s=16, alpha=0.55,
                color=STN_C, edgecolors="none", zorder=2)
    axL.scatter([xi], [np.mean(vals)], marker="D", s=46, color="#0B3D63",
                zorder=4, label="mean" if xi == 0 else None)
    matched = sum(1 for g in vals if g == 0)
    axL.annotate(f"{matched}/{len(vals)}\nexact", (xi, -14), ha="center",
                 fontsize=8, color="#0B3D63")
axL.axhline(0, color="0.6", lw=0.8, ls=":")
axL.set_xticks(range(len(ns))); axL.set_xticklabels(ns)
axL.set_xlabel(r"agent count $n$")
axL.set_ylabel("optimality gap (%)")
axL.set_title("Gap to executable optimum")
axL.set_ylim(-22, None)
axL.grid(True, axis="y", alpha=0.25, lw=0.5)
axL.legend(loc="upper left", framealpha=0.9)

# RIGHT: runtime STN vs ILP, log scale
xs = ns
stn_means = [np.mean(stn_rt[n]) for n in ns]
ilp_means = [np.mean(ilp_rt[n]) for n in ns]
axR.plot(xs, stn_means, "-s", color=STN_C, lw=2, ms=6, label="SlackCertify")
axR.plot(xs, ilp_means, "--o", color=ILP_C, lw=2, ms=6, label="ILP (exact)")
axR.set_yscale("log")
axR.set_xticks(ns); axR.set_xticklabels(ns)
axR.set_xlabel(r"agent count $n$")
axR.set_ylabel("certifier runtime (ms, log)")
axR.set_title("Runtime")
axR.grid(True, which="both", alpha=0.25, lw=0.5)
axR.legend(loc="upper left", framealpha=0.9)
# annotate speedup at n=20
if ns:
    n_last = ns[-1]
    sp = ilp_means[-1] / max(stn_means[-1], 1e-9)
    axR.annotate(f"{sp:.0f}$\\times$", (n_last, ilp_means[-1]),
                 textcoords="offset points", xytext=(-6, 6),
                 fontsize=10, color=ILP_C, ha="right")

fig.tight_layout(pad=0.6)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"wrote {OUT}")

print("\n--- summary ---")
allg = [g for n in ns for g in gaps_by_n[n]]
print(f"overall matched(0%): {sum(g==0 for g in allg)}/{len(allg)}  mean {np.mean(allg):.2f}%  median {np.median(allg):.2f}%")
for n in ns:
    v = gaps_by_n[n]
    print(f"  n={n}: mean {np.mean(v):.2f}%  max {max(v):.2f}%  STN {np.mean(stn_rt[n]):.2f}ms  ILP {np.mean(ilp_rt[n]):.1f}ms")
