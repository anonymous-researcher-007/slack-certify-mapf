"""Shared helpers for the plot generators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.plots.style import apply_paper_style, one_column_fig

__all__ = [
    "ROLLOUTS_PER_CELL_DEFAULT",
    "load_csv",
    "save_placeholder_figure",
    "successful_rows",
]


# Default Phase 7 K — every runner uses K=500 for paper-grade runs and
# K=50 for smokes. Used by Wilson-CI helpers when the CSV doesn't carry
# the rollout count explicitly.
ROLLOUTS_PER_CELL_DEFAULT: int = 500


def load_csv(path: Path) -> pd.DataFrame:
    """Read a Phase 7 CSV. Returns an empty DataFrame if the file is missing."""
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ``df`` to rows whose ``status`` is ``"ok"``.

    Returns an empty DataFrame when the column is missing.
    """
    if df.empty or "status" not in df.columns:
        return df.iloc[0:0]
    return df[df["status"] == "ok"].copy()


def save_placeholder_figure(output_pdf: Path, message: str) -> None:
    """Render a single-axes placeholder figure with ``message`` and save it.

    Plot generators call this when their input CSV is empty or contains
    no successful cells, so the pipeline always produces a PDF at the
    expected path rather than silently skipping a deliverable.
    """
    apply_paper_style()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with one_column_fig() as (fig, ax):
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="dimgray",
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.savefig(output_pdf)
