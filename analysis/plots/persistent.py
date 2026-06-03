"""RQ5 persistent-delay figure — nominal vs. probabilistic SlackCertify."""

from __future__ import annotations

from pathlib import Path

from analysis.plots._common import (
    ROLLOUTS_PER_CELL_DEFAULT,
    load_csv,
    save_placeholder_figure,
    successful_rows,
)
from analysis.plots.style import PAPER_PALETTE, two_column_fig
from analysis.statistics.wilson import wilson_ci

__all__ = ["generate_persistent_figure"]


def generate_persistent_figure(rq5_csv: Path, output_pdf: Path) -> None:
    """Render the RQ5 persistent-delay two-panel figure.

    Left panel: success rate vs. agent count, one line per method.
    Right panel: realised SoC overhead vs. agent count, one line per
    method (mean across seeds, with seed-level scatter underlay).
    """
    rq5_csv = Path(rq5_csv)
    output_pdf = Path(output_pdf)
    df = load_csv(rq5_csv)
    ok = successful_rows(df)
    if ok.empty:
        save_placeholder_figure(output_pdf, f"no successful rows in {rq5_csv.name}")
        return

    methods = sorted(ok["method"].unique())
    method_colors = dict(zip(methods, PAPER_PALETTE, strict=False))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with two_column_fig(n_panels=2) as (fig, axes):
        ax_sr, ax_soc = axes[0], axes[1]
        for method in methods:
            sub = ok[ok["method"] == method].sort_values("n_agents").copy()
            if sub.empty:
                continue
            grouped = (
                sub.groupby("n_agents")
                .agg(
                    success_rate=("success_rate", "mean"),
                    executed_soc_mean=("executed_soc_mean", "mean"),
                    n=("success_rate", "size"),
                )
                .reset_index()
            )
            xs = grouped["n_agents"].to_numpy()
            srs = grouped["success_rate"].to_numpy()
            soc = grouped["executed_soc_mean"].to_numpy()
            color = method_colors.get(method, "black")
            los, his = [], []
            for _, r in grouped.iterrows():
                n_total = int(r["n"]) * ROLLOUTS_PER_CELL_DEFAULT
                k = int(round(float(r["success_rate"]) * n_total))
                lo, hi = wilson_ci(k, max(1, n_total))
                los.append(lo)
                his.append(hi)
            ax_sr.plot(
                xs,
                srs,
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=color,
                label=method,
            )
            ax_sr.fill_between(xs, los, his, alpha=0.18, color=color, linewidth=0)
            ax_soc.plot(
                xs,
                soc,
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=color,
                label=method,
            )
        ax_sr.set_xlabel("agents")
        ax_sr.set_ylabel("success rate")
        ax_sr.set_ylim(-0.02, 1.02)
        ax_soc.set_xlabel("agents")
        ax_soc.set_ylabel("realised SoC (mean)")
        handles, labels = ax_sr.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(len(labels), 4),
                bbox_to_anchor=(0.5, 1.05),
                frameon=False,
            )
        fig.savefig(output_pdf)
