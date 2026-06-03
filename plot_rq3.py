"""Fig. 3 - RQ3 certificate tightness.
Empirical collision rate vs. target epsilon, with the y=eps diagonal
(certificate-holds boundary) and the certified risk upper bound shown as
the conservatism band. Points below the diagonal => certificate holds.
Reads results/rq3_tightness.csv. Saves figures/fig3_rq3_tightness.png 300dpi.
"""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/rq3_tightness.csv"
OUT = "figures/fig3_rq3_tightness.png"
DPI = 300
WARE_C, RAND_C = "#1F6FB2", "#C0392B"
RISK_C = "#7F8C8D"

rows = list(csv.DictReader(open(CSV)))
EPS = sorted({float(r["target_eps"]) for r in rows})

# gather ok cells per map
by_map = {"warehouse-10-20-10-2-1": collections.defaultdict(lambda: {"emp": [], "risk": []}),
          "random-32-32-20":        collections.defaultdict(lambda: {"emp": [], "risk": []})}
for r in rows:
    if r["status"] != "ok" or not r["risk_ub"]:
        continue
    e = float(r["target_eps"])
    by_map[r["map"]][e]["emp"].append(float(r["empirical_collision"]))
    by_map[r["map"]][e]["risk"].append(float(r["risk_ub"]))

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "axes.linewidth": 0.9,
})
fig, ax = plt.subplots(figsize=(6.2, 4.6))

# y = eps diagonal (certificate-holds boundary)
lo, hi = 0.0, max(EPS) * 1.12
ax.plot([lo, hi], [lo, hi], "--", color="0.35", lw=1.3, zorder=1,
        label=r"$y=\varepsilon$ (target)")
ax.fill_between([lo, hi], [lo, hi], hi, color="0.92", zorder=0)  # above diagonal = violates
ax.text(hi*0.97, hi*0.90, "certificate\nviolated", ha="right", va="top",
        fontsize=9, color="0.55", style="italic")

def _series(mp, color, label, dx):
    xs_pts, emp_pts = [], []
    risk_mean, emp_mean = [], []
    for e in EPS:
        d = by_map[mp][e]
        if not d["emp"]:
            risk_mean.append(np.nan); emp_mean.append(np.nan); continue
        jit = (np.random.RandomState(int(e*1000)).rand(len(d["emp"]))-0.5)*0.012
        xs_pts += list(np.full(len(d["emp"]), e) + dx + jit)
        emp_pts += d["emp"]
        risk_mean.append(np.mean(d["risk"])); emp_mean.append(np.mean(d["emp"]))
    ax.scatter(xs_pts, emp_pts, s=12, alpha=0.30, color=color, edgecolors="none", zorder=2)
    ax.plot([e+dx for e in EPS], emp_mean, "-o", color=color, lw=2, ms=6,
            zorder=4, label=f"{label}: empirical")
    ax.plot([e+dx for e in EPS], risk_mean, ":s", color=color, lw=1.6, ms=5,
            alpha=0.8, zorder=3, label=f"{label}: certified bound")

_series("warehouse-10-20-10-2-1", WARE_C, "warehouse", -0.004)
_series("random-32-32-20", RAND_C, "random-32-32-20", +0.004)

ax.set_xlabel(r"target collision bound $\varepsilon$")
ax.set_ylabel("collision rate")
ax.set_title("Certificate tightness: empirical vs. target")
ax.set_xlim(0.02, hi)
ax.set_ylim(0.0, hi)
ax.set_xticks(EPS); ax.set_xticklabels([str(e) for e in EPS])
ax.grid(True, alpha=0.25, lw=0.5)
ax.legend(loc="upper left", framealpha=0.93, fontsize=9.5)

fig.tight_layout(pad=0.6)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"wrote {OUT}")

# summary
print("\n--- summary ---")
import statistics as st
ok = [r for r in rows if r["status"]=="ok" and r["risk_ub"]]
viol = sum(1 for r in ok if float(r["empirical_collision"])>float(r["target_eps"]))
print(f"ok cells: {len(ok)}, empirical>eps violations: {viol}")
for e in EPS:
    we=by_map['warehouse-10-20-10-2-1'][e]; re=by_map['random-32-32-20'][e]
    allemp=we['emp']+re['emp']; allrisk=we['risk']+re['risk']
    if allemp:
        print(f"  eps={e}: emp {st.mean(allemp):.4f}  risk_ub {st.mean(allrisk):.4f}  "
              f"conservatism {st.mean(allrisk)/max(st.mean(allemp),1e-9):.1f}x  (n={len(allemp)})")
