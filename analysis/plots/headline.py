"""RQ1 headline figure — success rate vs. agent count, per map.

Reads ``results/raw/rq1_headline.csv`` and renders a two-panel
figure (one panel per map) with one line per method and Wilson-CI
shaded bands. Saves at IEEE two-column width × 2.5 in.
"""

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

__all__ = ["generate_headline_figure"]


def generate_headline_figure(rq1_csv: Path, output_pdf: Path) -> None:
    """Render the RQ1 headline figure.

    Parameters
    ----------
    rq1_csv
        CSV produced by ``RQ1HeadlineRunner``. Missing or empty CSV
        produces a placeholder figure (so the pipeline never fails
        silently on missing inputs).
    output_pdf
        Destination PDF path. Parent directories are created.
    """
    rq1_csv = Path(rq1_csv)
    output_pdf = Path(output_pdf)
    df = load_csv(rq1_csv)
    ok = successful_rows(df)
    if ok.empty:
        save_placeholder_figure(
            output_pdf,
            f"no successful rows in {rq1_csv.name}",
        )
        return

    maps = sorted(ok["map_name"].unique())
    methods = sorted(ok["method"].unique())
    method_colors = dict(zip(methods, PAPER_PALETTE, strict=False))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    n_panels = max(1, len(maps))
    with two_column_fig(n_panels=n_panels) as (fig, axes):
        for ax, map_name in zip(axes, maps, strict=False):
            sub = ok[ok["map_name"] == map_name]
            for method in methods:
                rows = sub[sub["method"] == method].sort_values("n_agents").copy()
                if rows.empty:
                    continue
                # Recompute Wilson CI here so the figure does not rely on
                # the runner having written ci_lo/ci_hi columns.
                xs = rows["n_agents"].to_numpy()
                ys = rows["success_rate"].to_numpy()
                los: list[float] = []
                his: list[float] = []
                for _, r in rows.iterrows():
                    k = int(round(float(r["success_rate"]) * ROLLOUTS_PER_CELL_DEFAULT))
                    lo, hi = wilson_ci(k, ROLLOUTS_PER_CELL_DEFAULT)
                    los.append(lo)
                    his.append(hi)
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    color=method_colors.get(method, "black"),
                    label=method,
                )
                ax.fill_between(
                    xs,
                    los,
                    his,
                    alpha=0.18,
                    color=method_colors.get(method, "black"),
                    linewidth=0,
                )
            ax.set_title(map_name)
            ax.set_xlabel("agents")
            ax.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("success rate")
        # One shared legend at the top of the figure.
        handles, labels = axes[0].get_legend_handles_labels()
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
