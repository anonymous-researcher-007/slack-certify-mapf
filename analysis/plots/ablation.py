"""Ablation figures — one PDF per ablation runner.

Each ablation produces a small bar / strip plot with one bar per method
and either total_waits or success_rate on the y-axis. The solver
ablation in the sandbox produces all-binary_missing rows; the generator
falls back to a placeholder for that case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.plots._common import (
    load_csv,
    save_placeholder_figure,
    successful_rows,
)
from analysis.plots.style import PAPER_PALETTE, one_column_fig

__all__ = ["generate_ablation_figures"]


def _bar_per_method(
    csv_path: Path,
    output_pdf: Path,
    title: str,
    y_field: str,
    y_label: str,
) -> Path:
    """Render a single ablation bar chart at ``output_pdf`` and return its path."""
    df = load_csv(csv_path)
    ok = successful_rows(df)
    if ok.empty or y_field not in ok.columns:
        save_placeholder_figure(output_pdf, f"no successful rows in {csv_path.name}")
        return output_pdf

    grouped = (
        ok.groupby("method")[y_field]
        .agg(["mean", "std", "size"])
        .reset_index()
        .sort_values("method")
    )
    methods = grouped["method"].tolist()
    means = grouped["mean"].to_numpy()
    stds = grouped["std"].fillna(0.0).to_numpy()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with one_column_fig() as (fig, ax):
        x = np.arange(len(methods))
        ax.bar(
            x,
            means,
            color=[PAPER_PALETTE[i % len(PAPER_PALETTE)] for i in range(len(methods))],
            edgecolor="black",
            linewidth=0.4,
            yerr=stds,
            capsize=2,
            error_kw={"linewidth": 0.6, "ecolor": "black"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        fig.savefig(output_pdf)
    return output_pdf


def generate_ablation_figures(
    ablation_ordering_csv: Path,
    ablation_solver_csv: Path,
    ablation_budget_csv: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Render one figure per ablation; return ``{name: pdf_path}``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    out["ordering"] = _bar_per_method(
        Path(ablation_ordering_csv),
        output_dir / "ablation_ordering.pdf",
        title="Conflict ordering ablation",
        y_field="total_waits",
        y_label="total waits (mean)",
    )
    out["solver"] = _bar_per_method(
        Path(ablation_solver_csv),
        output_dir / "ablation_solver.pdf",
        title="Solver ablation",
        y_field="total_waits",
        y_label="total waits (mean)",
    )
    out["budget"] = _bar_per_method(
        Path(ablation_budget_csv),
        output_dir / "ablation_budget.pdf",
        title="Budget allocator ablation",
        y_field="total_waits",
        y_label="total waits (mean)",
    )
    return out
