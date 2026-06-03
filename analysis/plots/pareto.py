"""RQ2 Pareto figure — SoC overhead vs. success rate per (method, delta)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.plots._common import (
    load_csv,
    save_placeholder_figure,
    successful_rows,
)
from analysis.plots.style import PAPER_PALETTE, one_column_fig

__all__ = ["generate_pareto_figure"]


# Distinct markers per method (cycled if more methods than markers).
_METHOD_MARKERS: dict[str, str] = {
    "slack_certify_bounded": "o",
    "kr_eecbs": "s",
    "kr_pbs": "^",
}


def generate_pareto_figure(rq2_csv: Path, output_pdf: Path) -> None:
    """Render the RQ2 Pareto scatter of SoC overhead vs. success rate.

    Each marker is one cell from the CSV. Marker shape encodes the
    method; colour shading encodes the bounded-delta margin (``delta``
    column; light → small Δ, dark → large Δ). Cells with
    ``status="binary_missing"`` are dropped (the Phase 7 convention).
    """
    rq2_csv = Path(rq2_csv)
    output_pdf = Path(output_pdf)
    df = load_csv(rq2_csv)
    ok = successful_rows(df)
    if ok.empty:
        save_placeholder_figure(output_pdf, f"no successful rows in {rq2_csv.name}")
        return

    if "executed_soc_mean" not in ok.columns or "delta" not in ok.columns:
        save_placeholder_figure(
            output_pdf,
            f"{rq2_csv.name} missing executed_soc_mean / delta",
        )
        return

    # SoC overhead. Preferred path (true metric M2): when the logged
    # ``nominal_soc`` column SoC(π) is present and positive, compute
    # ``executed_soc_mean / nominal_soc - 1`` directly — the
    # numerator/denominator share the sum-of-arrival-times SoC definition.
    # Fallback (legacy proxy): older CSVs without ``nominal_soc`` use the
    # minimum executed SoC across methods per (map, n, scen_seed) as the
    # reference; on a single-method CSV that collapses to the row itself,
    # producing 0 overhead.
    if "nominal_soc" in ok.columns and ok["nominal_soc"].gt(0.0).any():
        merged = ok.copy()
        denom = merged["nominal_soc"].where(merged["nominal_soc"] > 0.0, np.nan)
        merged["soc_overhead"] = merged["executed_soc_mean"] / denom - 1.0
        merged["soc_overhead"] = merged["soc_overhead"].fillna(0.0)
    else:
        ref_keys = ["map_name", "n_agents", "scen_seed"]
        ref = ok.groupby(ref_keys)["executed_soc_mean"].min().rename("soc_ref").reset_index()
        merged = ok.merge(ref, on=ref_keys, how="left")
        merged["soc_overhead"] = (merged["executed_soc_mean"] - merged["soc_ref"]) / merged[
            "soc_ref"
        ].replace(0.0, np.nan)
        merged["soc_overhead"] = merged["soc_overhead"].fillna(0.0)

    deltas = sorted(merged["delta"].dropna().unique())
    methods = sorted(merged["method"].unique())
    method_markers = {
        m: _METHOD_MARKERS.get(m, ["o", "s", "^", "D", "v", "P"][i % 6])
        for i, m in enumerate(methods)
    }

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with one_column_fig() as (fig, ax):
        # Colour shade by delta — light (small Δ) → dark (large Δ).
        for method in methods:
            marker = method_markers[method]
            for i, d in enumerate(deltas):
                sub = merged[(merged["method"] == method) & (merged["delta"] == d)]
                if sub.empty:
                    continue
                shade = PAPER_PALETTE[(i + 2) % len(PAPER_PALETTE)]
                ax.scatter(
                    sub["soc_overhead"],
                    sub["success_rate"],
                    marker=marker,
                    color=shade,
                    s=22,
                    alpha=0.85,
                    label=f"{method} (Δ={int(d)})",
                    edgecolors="black",
                    linewidths=0.4,
                )
        ax.set_xlabel("SoC overhead (relative)")
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(
            loc="lower right",
            fontsize=6.5,
            frameon=False,
            ncol=1,
        )
        fig.savefig(output_pdf)
