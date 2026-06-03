"""RQ4 optimality-gap figure — gap distribution + matched-optimum rate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.plots._common import (
    load_csv,
    save_placeholder_figure,
    successful_rows,
)
from analysis.plots.style import PAPER_PALETTE, two_column_fig

__all__ = ["generate_optimality_gap_figure"]


def _parse_diag(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_gap_pct(diag: dict[str, Any]) -> float | None:
    """Pull the labelled gap_pct out of the slack_certify_bounded diag."""
    for key in ("gap_pct_to_optimum", "gap_pct_to_feasible", "gap_pct"):
        if key in diag and diag[key] is not None:
            try:
                return float(diag[key])
            except (TypeError, ValueError):
                return None
    return None


def generate_optimality_gap_figure(rq4_csv: Path, output_pdf: Path) -> None:
    """Render the RQ4 optimality-gap figure.

    Left panel: boxplot of gap_pct (greedy-vs-ILP) per agent count.
    Right panel: bar chart of the fraction of cells where greedy
    matched the proven ILP optimum, per agent count. Rows where the
    ILP didn't prove optimality (``ilp_proven_optimal == False``) are
    excluded from the matched-optimum panel.
    """
    rq4_csv = Path(rq4_csv)
    output_pdf = Path(output_pdf)
    df = load_csv(rq4_csv)
    ok = successful_rows(df)
    if ok.empty or "method" not in ok.columns:
        save_placeholder_figure(output_pdf, f"no successful rows in {rq4_csv.name}")
        return

    sb = ok[ok["method"] == "slack_certify_bounded"].copy()
    if sb.empty:
        save_placeholder_figure(
            output_pdf,
            f"{rq4_csv.name}: no slack_certify_bounded rows with status=ok",
        )
        return

    sb["_diag"] = sb["diagnostics"].apply(_parse_diag)
    sb["_gap_pct"] = sb["_diag"].apply(_extract_gap_pct)
    sb["_matched"] = sb["_diag"].apply(lambda d: d.get("matched_optimum"))
    sb["_proven"] = sb["_diag"].apply(lambda d: bool(d.get("ilp_proven_optimal", False)))

    counts = sorted(sb["n_agents"].unique())
    if not counts:
        save_placeholder_figure(output_pdf, "no agent_count groups to plot")
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with two_column_fig(n_panels=2) as (fig, axes):
        # Left: gap_pct boxplot per agent count.
        box_data: list[np.ndarray] = []
        labels: list[str] = []
        for n in counts:
            vals = sb.loc[sb["n_agents"] == n, "_gap_pct"].dropna().to_numpy()
            box_data.append(vals if vals.size else np.array([0.0]))
            labels.append(str(int(n)))
        bp = axes[0].boxplot(
            box_data,
            labels=labels,
            showmeans=True,
            patch_artist=True,
            widths=0.6,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(PAPER_PALETTE[2])
            patch.set_alpha(0.5)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        axes[0].set_xlabel("agents")
        axes[0].set_ylabel("gap pct (greedy − ILP)")
        axes[0].axhline(0.0, color="black", linewidth=0.6, linestyle="--")

        # Right: matched-optimum rate per agent count (proven-optimal only).
        matched_rate: list[float] = []
        for n in counts:
            proven = sb[(sb["n_agents"] == n) & (sb["_proven"])]
            if proven.empty:
                matched_rate.append(0.0)
                continue
            frac = float(proven["_matched"].astype(bool).mean())
            matched_rate.append(frac)
        axes[1].bar(
            range(len(counts)),
            matched_rate,
            color=PAPER_PALETTE[3],
            edgecolor="black",
            linewidth=0.4,
        )
        axes[1].set_xticks(range(len(counts)))
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("agents")
        axes[1].set_ylabel("greedy matched ILP optimum")
        axes[1].set_ylim(0.0, 1.05)
        fig.savefig(output_pdf)
