"""Tab 2 — success rate per (map × method) for the close-analogue baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.statistics.wilson import wilson_ci
from analysis.tables._common import (
    booktabs_footer,
    booktabs_header,
    escape_latex,
    format_ci,
    load_csv,
    successful_rows,
    wrap_in_tabular,
    write_tex,
)

__all__ = ["generate_analogues_table"]


# Column label → method string in the rq3 CSV.
_COLUMNS: dict[str, str] = {
    "Nominal": "nominal_unhardened",
    "BTPG-max": "btpg_max",
    "Kottinger-offline": "kottinger_offline",
    "Ours": "slack_certify_probabilistic",
}


# K used by Phase 7 rollouts. The runner doesn't carry the per-row
# rollout count, so we approximate Wilson bounds with the canonical
# default — close to the empirical CI columns the runner writes.
_K_DEFAULT = 500


def _aggregate(rows: pd.DataFrame) -> tuple[float, float, float] | None:
    if rows.empty:
        return None
    sr = float(rows["success_rate"].mean())
    n_total = int(len(rows)) * _K_DEFAULT
    k = max(0, min(n_total, int(round(sr * n_total))))
    lo, hi = wilson_ci(k, max(1, n_total))
    return sr, lo, hi


def generate_analogues_table(rq3_csv: Path, output_tex: Path) -> None:
    """Render Tab 2: success rate per map × method (Wilson CI)."""
    ok = successful_rows(load_csv(Path(rq3_csv)))
    columns = list(_COLUMNS.keys())

    if ok.empty:
        body = booktabs_header("Map", *columns)
        body += (
            "\\multicolumn{"
            + str(1 + len(columns))
            + "}{c}{\\textit{no successful rows in rq3 CSV}} \\\\\n"
        )
        body += booktabs_footer()
        write_tex(wrap_in_tabular(body, n_cols=1 + len(columns)), Path(output_tex))
        return

    maps = sorted(ok["map_name"].unique())
    body_lines: list[str] = []
    for map_name in maps:
        cells: list[str] = [escape_latex(map_name)]
        for col in columns:
            method = _COLUMNS[col]
            rows = ok[(ok["map_name"] == map_name) & (ok["method"] == method)]
            agg = _aggregate(rows)
            cells.append("--" if agg is None else format_ci(*agg, digits=2))
        body_lines.append(" & ".join(cells) + " \\\\")

    header = booktabs_header("Map", *columns)
    body = header + "\n".join(body_lines) + "\n" + booktabs_footer()
    tex = wrap_in_tabular(body, n_cols=1 + len(columns))
    write_tex(tex, Path(output_tex))
