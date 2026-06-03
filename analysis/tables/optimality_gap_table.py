"""Tab 3 — optimality gap of greedy SlackCertify against the proven ILP optimum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.tables._common import (
    booktabs_footer,
    booktabs_header,
    format_pct,
    load_csv,
    wrap_in_tabular,
    write_tex,
)

__all__ = ["generate_optimality_gap_table"]


def _parse_diag(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    if not isinstance(value, (str, bytes, bytearray)):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def generate_optimality_gap_table(rq4_csv: Path, output_tex: Path) -> None:
    """Render Tab 3: per-n cells solved, matched optimum, mean/max gap, ILP timeouts.

    Cell semantics for this table are restricted to **proven-optimal**
    ILP outcomes (``diagnostics.ilp_proven_optimal == True``). The
    count of feasible-only rows (``gap_pct_to_feasible``) is reported
    in a LaTeX footnote so reviewers see the proven-vs-feasible split
    without confusing them in the headline numbers.
    """
    df = load_csv(Path(rq4_csv))
    columns = [
        "$n$",
        "Cells solved",
        "Matched optimum",
        "Mean gap",
        "Max gap",
        "ILP timeouts",
    ]
    if df.empty:
        body = booktabs_header(*columns)
        body += "\\multicolumn{" + str(len(columns)) + "}{c}{\\textit{no rows in rq4 CSV}} \\\\\n"
        body += booktabs_footer()
        write_tex(wrap_in_tabular(body, n_cols=len(columns)), Path(output_tex))
        return

    sb = df[df["method"] == "slack_certify_bounded"].copy()
    ilp = df[df["method"] == "ilp_exact"].copy()

    sb["_diag"] = sb["diagnostics"].apply(_parse_diag)
    sb["_proven"] = sb["_diag"].apply(lambda d: bool(d.get("ilp_proven_optimal", False)))
    sb["_matched"] = sb["_diag"].apply(lambda d: d.get("matched_optimum"))
    sb["_gap_pct_to_optimum"] = sb["_diag"].apply(lambda d: d.get("gap_pct_to_optimum"))
    sb["_gap_pct_to_feasible"] = sb["_diag"].apply(lambda d: d.get("gap_pct_to_feasible"))

    n_values = sorted(int(n) for n in sb["n_agents"].dropna().unique())
    body_lines: list[str] = []
    feasible_only_total = 0
    for n in n_values:
        sub = sb[sb["n_agents"] == n]
        proven = sub[sub["_proven"]]
        feasible_only = sub[(~sub["_proven"]) & sub["_gap_pct_to_feasible"].notna()]
        feasible_only_total += int(len(feasible_only))
        n_solved = int((sub["status"] == "ok").sum())
        # Matched-optimum rate over proven-optimal cells only.
        if proven.empty:
            matched_pct_str = "--"
            mean_gap_str = "--"
            max_gap_str = "--"
        else:
            matched_frac = float(proven["_matched"].astype(bool).mean())
            matched_pct_str = format_pct(matched_frac, digits=1)
            gaps = proven["_gap_pct_to_optimum"].dropna().astype(float).to_numpy()
            if gaps.size:
                mean_gap_str = format_pct(float(np.mean(gaps)) / 100.0, digits=1)
                max_gap_str = format_pct(float(np.max(gaps)) / 100.0, digits=1)
            else:
                mean_gap_str = "--"
                max_gap_str = "--"
        # ILP-side timeouts at this n.
        n_timeouts = int(((ilp["n_agents"] == n) & (ilp["status"] == "timeout")).sum())
        body_lines.append(
            " & ".join(
                [
                    str(n),
                    str(n_solved),
                    matched_pct_str,
                    mean_gap_str,
                    max_gap_str,
                    str(n_timeouts),
                ]
            )
            + " \\\\"
        )

    if not body_lines:
        body_lines.append(
            "\\multicolumn{" + str(len(columns)) + "}{c}{\\textit{no per-$n$ rows in rq4 CSV}} \\\\"
        )

    body = booktabs_header(*columns) + "\n".join(body_lines) + "\n"
    body += booktabs_footer()
    tex = wrap_in_tabular(body, n_cols=len(columns))
    # Footnote: how many feasible-only ILP rows are excluded from the
    # headline matched-optimum percentage.
    footnote = (
        f"\\par\\smallskip\n\\footnotesize "
        f"Reported gaps cover proven-optimal ILP outcomes only. "
        f"Across all $n$, {feasible_only_total} additional cell"
        f"{'s' if feasible_only_total != 1 else ''} returned a "
        f"feasible-only ILP upper bound (gap unprovable).\n"
    )
    write_tex(tex + footnote, Path(output_tex))
