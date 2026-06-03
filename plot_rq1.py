"""Fig. 1 - RQ1 headline robustness: success rate vs. agent count, two maps,
nominal vs. SlackCertify-bounded, with CIs and infeasibility marking.
No external LaTeX required. Saves a high-res PNG into figures/.
"""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/rq1_bounded.csv"
OUT = "figures/fig1_rq1_headline.png"
DPI = 300  # publication raster resolution

COL_MAP, COL_N, COL_METHOD = "map_name", "n_agents", "method"
COL_SR, COL_STATUS = "success_rate", "status"

METHODS = {
    "nominal_unhardened":    dict(label="Un-hardened nominal", color="#C0392B", marker="o", ls="--"),
    "slack_certify_bounded": dict(label=r"SlackCertify ($\Delta=2$)", color="#1F6FB2", marker="s", ls="-"),
}
MAP_ORDER = ["warehouse-10-20-10-2-1", "random-32-32-20"]
INFEAS_COLOR = "#1F6FB2"

rows = list(csv.DictReader(open(CSV)))
succ = collections.defaultdict(list)
infeas = collections.defaultdict(lambda: [0, 0])
for r in rows:
    m, n, meth, st = r[COL_MAP], int(r[COL_N]), r[COL_METHOD], r[COL_STATUS]
    if meth == "slack_certify_bounded":
        infeas[(m, n)][1] += 1
        if st == "wait_infeasible":
            infeas[(m, n)][0] += 1
    if st == "ok":
        succ[(m, n, meth)].append(float(r[COL_SR]))

all_n = sorted({int(r[COL_N]) for r in rows})

# Paper-suitable typography: serif, larger sizes, clean math via mathtext.
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
})

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)

for ax, m in zip(axes, MAP_ORDER):
    for meth, sty in METHODS.items():
        xs, ys, lo, hi = [], [], [], []
        for n in all_n:
            vals = succ.get((m, n, meth), [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                se = sd / len(vals) ** 0.5
                clo, chi = max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)
            else:
                clo, chi = mean, mean
            xs.append(n); ys.append(mean * 100)
            lo.append((mean - clo) * 100); hi.append((chi - mean) * 100)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], color=sty["color"], marker=sty["marker"],
                        ls=sty["ls"], capsize=3, capthick=1.2, label=sty["label"], zorder=3)
    # infeasibility marking
    for n in all_n:
        ni, tot = infeas.get((m, n), [0, 0])
        if tot > 0 and ni == tot:           # fully infeasible: no bounded point
            ax.scatter([n], [1.0], marker="x", color=INFEAS_COLOR, s=70, linewidths=2.0, zorder=4)
            ax.annotate("infeasible", (n, 6), color=INFEAS_COLOR, fontsize=10,
                        ha="center", style="italic")
        elif tot > 0 and ni > 0:            # partial: small note
            ax.annotate(f"{ni}/{tot} inf.", (n, 103.5), color=INFEAS_COLOR, fontsize=9, ha="center")

    ax.set_title(m, fontsize=12, family="monospace")
    ax.set_xlabel(r"agent count $n$")
    ax.set_xscale("log"); ax.set_xticks(all_n); ax.set_xticklabels(all_n)
    ax.set_ylim(-4, 109)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.tick_params(direction="out", length=4)

axes[0].set_ylabel("success rate (\\%)" if plt.rcParams["text.usetex"] else "success rate (%)")
axes[0].legend(loc="lower left", framealpha=0.95, edgecolor="0.7")
fig.tight_layout(pad=0.6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"wrote {OUT} at {DPI} dpi")

print("\n--- plotted means (%) ---")
for m in MAP_ORDER:
    for n in all_n:
        b = succ.get((m, n, 'slack_certify_bounded'), [])
        nom = succ.get((m, n, 'nominal_unhardened'), [])
        ni, tot = infeas.get((m, n), [0, 0])
        bs = f"{sum(b)/len(b)*100:6.1f}" if b else "   inf"
        ns = f"{sum(nom)/len(nom)*100:6.1f}" if nom else "   nan"
        print(f"{m:24} n={n:<4} nominal={ns}  bounded={bs}  infeas={ni}/{tot}")
