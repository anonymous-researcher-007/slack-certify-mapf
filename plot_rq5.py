"""Fig. 5 - RQ5 persistent-delay stress: P2 (i.i.d.) vs P3 (correlated) success
per trigger probability, matched marginal rate, with Wilson CIs.
Reads results/rq5_persistent.csv. Saves figures/fig5_rq5_persistent.png 300dpi.
"""
import collections
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/rq5_persistent.csv"
OUT = "figures/fig5_rq5_persistent.png"
DPI = 300
P2_C, P3_C = "#1F6FB2", "#C0392B"

rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "ok"]
PTS = sorted({float(r["p_trigger"]) for r in rows})

def wilson(p, n, z=1.96):
    if n == 0: return (p, p)
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

# pooled success per trigger; CI from per-instance Bernoulli mean
agg = collections.defaultdict(lambda: {"p2": [], "p3": []})
for r in rows:
    pt = float(r["p_trigger"])
    agg[pt]["p2"].append(float(r["success_P2"]))
    agg[pt]["p3"].append(float(r["success_P3"]))

p2m = [np.mean(agg[pt]["p2"]) for pt in PTS]
p3m = [np.mean(agg[pt]["p3"]) for pt in PTS]
# CI across instances (SEM-based), simpler & honest for a mean of rates
def ci(vals):
    np.mean(vals); s = np.std(vals, ddof=1)/math.sqrt(len(vals)) if len(vals) > 1 else 0
    return 1.96*s
p2e = [ci(agg[pt]["p2"]) for pt in PTS]
p3e = [ci(agg[pt]["p3"]) for pt in PTS]

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.linewidth": 0.9,
})
fig, ax = plt.subplots(figsize=(6.2, 4.2))
x = np.arange(len(PTS)); w = 0.38
ax.bar(x - w/2, p2m, w, yerr=p2e, capsize=3, color=P2_C, label="P2 (i.i.d. Bernoulli)")
ax.bar(x + w/2, p3m, w, yerr=p3e, capsize=3, color=P3_C, label="P3 (correlated stalls)")
ax.set_xticks(x); ax.set_xticklabels([str(pt) for pt in PTS])
ax.set_xlabel(r"persistent trigger probability $p_{\mathrm{trig}}$")
ax.set_ylabel("success rate")
ax.set_title("Persistent-delay stress (matched marginal rate)")
ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", alpha=0.25, lw=0.5)
ax.legend(loc="upper right", framealpha=0.93)
fig.tight_layout(pad=0.6)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"wrote {OUT}")
print("\n--- summary ---")
for i, pt in enumerate(PTS):
    print(f"  pt={pt}: P2={p2m[i]:.3f} P3={p3m[i]:.3f} ratio={p3m[i]/max(p2m[i],1e-9):.2f}")
