"""Tab 1 — median certifier runtime per (map, n) across methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.tables._common import (
    booktabs_footer,
    booktabs_header,
    escape_latex,
    load_csv,
    successful_rows,
    wrap_in_tabular,
    write_tex,
)

__all__ = ["generate_runtime_table"]


# Public column → list of (csv_alias, method_filter) pairs. Each
# alias maps to which CSV the row should be pulled from, and which
# ``method`` string within that CSV. The Ours column mixes both
# bounded and probabilistic SlackCertify so we can report a single
# headline runtime per (map, n).
_COLUMN_SOURCES: dict[str, list[tuple[str, str]]] = {
    "kR-EECBS": [("rq2", "kr_eecbs")],
    "BTPG-max": [("rq3", "btpg_max")],
    "Kottinger": [("rq3", "kottinger_offline")],
    "Ours": [
        ("rq1", "slack_certify_bounded"),
        ("rq1", "slack_certify_probabilistic"),
    ],
}


def _median_runtime(rows: pd.DataFrame) -> str:
    """Median certifier runtime in seconds, formatted to 2 decimal places."""
    if rows.empty or "certifier_runtime_s" not in rows.columns:
        return "--"
    vals = rows["certifier_runtime_s"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return "--"
    return f"{float(np.median(vals)):.2f}"


def generate_runtime_table(rq1_csv: Path, rq2_csv: Path, rq3_csv: Path, output_tex: Path) -> None:
    """Render Tab 1: median certifier runtime per (map, n) and method."""
    sources: dict[str, pd.DataFrame] = {
        "rq1": successful_rows(load_csv(Path(rq1_csv))),
        "rq2": successful_rows(load_csv(Path(rq2_csv))),
        "rq3": successful_rows(load_csv(Path(rq3_csv))),
    }
    # Build the (map, n) row set from the union of every source.
    pairs: set[tuple[str, int]] = set()
    for df in sources.values():
        if df.empty:
            continue
        for _, row in df[["map_name", "n_agents"]].drop_duplicates().iterrows():
            pairs.add((str(row["map_name"]), int(row["n_agents"])))
    rows_sorted = sorted(pairs)
    columns = list(_COLUMN_SOURCES.keys())

    body_lines: list[str] = []
    for map_name, n_agents in rows_sorted:
        cells: list[str] = [
            escape_latex(map_name),
            str(int(n_agents)),
        ]
        for col in columns:
            spec_list = _COLUMN_SOURCES[col]
            collected: list[pd.DataFrame] = []
            for alias, method in spec_list:
                df = sources[alias]
                if df.empty:
                    continue
                hits = df[
                    (df["map_name"] == map_name)
                    & (df["n_agents"] == n_agents)
                    & (df["method"] == method)
                ]
                if not hits.empty:
                    collected.append(hits)
            cell_rows = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
            cells.append(_median_runtime(cell_rows))
        body_lines.append(" & ".join(cells) + " \\\\")

    if not body_lines:
        # Empty placeholder so the paper section can still \input{} the
        # file without a TeX error.
        body_lines.append("\\multicolumn{" + str(2 + len(columns)) + "}{c}{\\textit{no data}} \\\\")

    header = booktabs_header(
        "Map",
        "$n$",
        *columns,
    )
    body = header + "\n".join(body_lines) + "\n" + booktabs_footer()
    tex = wrap_in_tabular(body, n_cols=2 + len(columns))
    write_tex(tex, Path(output_tex))
