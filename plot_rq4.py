import collections
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

r3 = list(csv.DictReader(open("results/rq3_tightness.csv")))
r1 = list(csv.DictReader(open("results/rq1_bounded.csv")))
WARE, RAND = "warehouse-10-20-10-2-1", "random-32-32-20"
WC, RC = "#1F6FB2", "#C0392B"
seps = collections.defaultdict(list)
for r in r3:
    if r["status"] == "ok" and r["sep_s"]:
        seps[(r["map"], int(r["n"]))].append(int(r["sep_s"]))
waits = collections.defaultdict(list); inf = collections.Counter(); tot = collections.Counter()
for r in r1:
    mp = r["map_name"]; n = int(r["n_agents"]); tot[(mp,n)] += 1
    if r["status"] == "wait_infeasible": inf[(mp,n)] += 1
    elif r["status"] == "ok" and r.get("total_waits") not in (None, ""):
        try: waits[(mp,n)].append(float(r["total_waits"]))
        except ValueError: pass
plt.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.9))
# LEFT: separation (solid) + waits (dashed, log) -- skip zero-mean waits points
wns = sorted({n for (m,n) in seps if m==WARE}); rns = sorted({n for (m,n) in seps if m==RAND})
axL.plot(wns, [np.median(seps[(WARE,n)]) for n in wns], "-o", color=WC, lw=2, ms=6, label="warehouse: sep. $s$")
axL.plot(rns, [np.median(seps[(RAND,n)]) for n in rns], "-o", color=RC, lw=2, ms=6, label="random: sep. $s$")
axL.set_xlabel(r"agent count $n$"); axL.set_ylabel(r"selected separation $s$ (median)")
axL.set_title("Certifier cost"); axL.grid(True, alpha=0.25, lw=0.5)
axL.set_xticks(sorted(set(wns+rns)))
axL2 = axL.twinx()
wwn = [n for n in sorted({k[1] for k in waits if k[0]==WARE}) if np.mean(waits[(WARE,n)])>0]
rwn = [n for n in sorted({k[1] for k in waits if k[0]==RAND}) if np.mean(waits[(RAND,n)])>0]
axL2.plot(wwn, [np.mean(waits[(WARE,n)]) for n in wwn], "--s", color=WC, lw=1.5, ms=5, alpha=0.7, label="warehouse: waits")
axL2.plot(rwn, [np.mean(waits[(RAND,n)]) for n in rwn], "--s", color=RC, lw=1.5, ms=5, alpha=0.7, label="random: waits")
axL2.set_ylabel("mean inserted waits", color="0.3"); axL2.set_yscale("log")
h1,l1 = axL.get_legend_handles_labels(); h2,l2 = axL2.get_legend_handles_labels()
axL.legend(h1+h2, l1+l2, loc="upper left", fontsize=8.5, framealpha=0.9)
# RIGHT: infeasibility
all_n = sorted({n for (m,n) in tot})
for m,c,lab in [(WARE,WC,"warehouse"),(RAND,RC,"random-32-32-20")]:
    ys = [100*inf[(m,n)]/tot[(m,n)] if tot.get((m,n)) else np.nan for n in all_n]
    axR.plot(all_n, ys, "-o", color=c, lw=2, ms=6, label=lab)
axR.set_xlabel(r"agent count $n$"); axR.set_ylabel("fixed-order infeasible (%)")
axR.set_title("Infeasibility (rotational deadlock)"); axR.set_xticks(all_n)
axR.set_ylim(-3, None); axR.grid(True, alpha=0.25, lw=0.5); axR.legend(loc="upper left", framealpha=0.9)
fig.tight_layout(pad=0.6); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig4_rq4_cost.png", dpi=300, bbox_inches="tight")
print("wrote two-panel figures/fig4_rq4_cost.png")
