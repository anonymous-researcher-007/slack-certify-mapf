"""RQ3 close-analogues figure — bar chart of success rate per map × method."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.plots._common import (
    ROLLOUTS_PER_CELL_DEFAULT,
    load_csv,
    save_placeholder_figure,
    successful_rows,
)
from analysis.plots.style import PAPER_PALETTE, one_column_fig
from analysis.statistics.wilson import wilson_ci

__all__ = ["generate_close_analogues_figure"]


def generate_close_analogues_figure(rq3_csv: Path, output_pdf: Path) -> None:
    """Render the RQ3 grouped bar chart.

    Groups by map; one bar per method per group. Wilson-CI error bars
    use the per-row aggregated rollout count when present, otherwise
    fall back to the K=500 default.
    """
    rq3_csv = Path(rq3_csv)
    output_pdf = Path(output_pdf)
    df = load_csv(rq3_csv)
    ok = successful_rows(df)
    if ok.empty:
        save_placeholder_figure(output_pdf, f"no successful rows in {rq3_csv.name}")
        return

    # Aggregate by (map, method) — mean success rate across seeds.
    grouped = ok.groupby(["map_name", "method"])["success_rate"].mean().reset_index()
    counts = ok.groupby(["map_name", "method"]).size().rename("n_cells").reset_index()
    grouped = grouped.merge(counts, on=["map_name", "method"])

    maps = sorted(grouped["map_name"].unique())
    methods = sorted(grouped["method"].unique())
    method_colors = dict(zip(methods, PAPER_PALETTE, strict=False))

    x = np.arange(len(maps), dtype=float)
    width = 0.8 / max(1, len(methods))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with one_column_fig() as (fig, ax):
        for i, method in enumerate(methods):
            heights: list[float] = []
            errs_lo: list[float] = []
            errs_hi: list[float] = []
            for map_name in maps:
                row = grouped[(grouped["map_name"] == map_name) & (grouped["method"] == method)]
                if row.empty:
                    heights.append(0.0)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                    continue
                sr = float(row["success_rate"].iloc[0])
                n_total = int(row["n_cells"].iloc[0]) * ROLLOUTS_PER_CELL_DEFAULT
                k = int(round(sr * n_total))
                lo, hi = wilson_ci(k, max(1, n_total))
                heights.append(sr)
                errs_lo.append(max(0.0, sr - lo))
                errs_hi.append(max(0.0, hi - sr))
            offsets = x + (i - (len(methods) - 1) / 2) * width
            ax.bar(
                offsets,
                heights,
                width=width,
                color=method_colors.get(method, "gray"),
                edgecolor="black",
                linewidth=0.4,
                label=method,
                yerr=[errs_lo, errs_hi],
                capsize=2,
                error_kw={"linewidth": 0.6, "ecolor": "black"},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(maps, rotation=20, ha="right")
        ax.set_ylabel("success rate")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7, frameon=False, ncol=min(3, max(1, len(methods))))
        fig.savefig(output_pdf)
