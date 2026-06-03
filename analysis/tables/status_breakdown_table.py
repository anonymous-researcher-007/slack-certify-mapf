"""Tab 4 — honest-reporting status breakdown per experiment × method.

For each Phase 7 CSV, aggregate the cell-level :attr:`CellResult.status`
counts and emit one row per (experiment, method) showing how many
cells landed in each terminal category. This is the headline
"honest reporting" table that §V uses to surface which methods
degrade gracefully versus catastrophically.
"""

from __future__ import annotations

from pathlib import Path

from analysis.tables._common import (
    STATUS_ABBREVIATIONS,
    booktabs_footer,
    booktabs_header,
    escape_latex,
    format_status_counts,
    load_csv,
    wrap_in_tabular,
    write_tex,
)

__all__ = ["generate_status_breakdown_table"]


def generate_status_breakdown_table(all_csvs: dict[str, Path], output_tex: Path) -> None:
    """Render Tab 4: status distribution per (experiment, method).

    Parameters
    ----------
    all_csvs
        Mapping of experiment label → CSV path. The label is used
        verbatim as the first-column value (escape-LaTeX'd).
    output_tex
        Output ``.tex`` path. Parent directories are created.
    """
    columns = ["Experiment", "Method", "Cells", "Breakdown"]
    body_lines: list[str] = []
    seen_statuses: set[str] = set()
    for experiment, csv_path in all_csvs.items():
        df = load_csv(Path(csv_path))
        if df.empty or "status" not in df.columns or "method" not in df.columns:
            continue
        for method in sorted(df["method"].dropna().unique()):
            rows = df[df["method"] == method]
            statuses = rows["status"].dropna().astype(str).tolist()
            seen_statuses.update(statuses)
            body_lines.append(
                " & ".join(
                    [
                        escape_latex(experiment),
                        escape_latex(method),
                        str(len(statuses)),
                        format_status_counts(statuses),
                    ]
                )
                + " \\\\"
            )

    if not body_lines:
        body = booktabs_header(*columns)
        body += "\\multicolumn{" + str(len(columns)) + "}{c}{\\textit{no rows in any CSV}} \\\\\n"
        body += booktabs_footer()
        write_tex(wrap_in_tabular(body, n_cols=len(columns)), Path(output_tex))
        return

    body = booktabs_header(*columns) + "\n".join(body_lines) + "\n"
    body += booktabs_footer()
    tex = wrap_in_tabular(body, n_cols=len(columns))
    # Legend footnote naming every abbreviation that actually appeared
    # plus any canonical ones we expect §V to reference. This lets
    # reviewers decode the "Breakdown" column without flipping back
    # to the appendix.
    legend_pairs: list[str] = []
    for canonical, abbr in STATUS_ABBREVIATIONS.items():
        if canonical in seen_statuses:
            legend_pairs.append(f"\\textbf{{{abbr}}}={escape_latex(canonical)}")
    legend = (
        f"\\par\\smallskip\n\\footnotesize Status legend: " f"{', '.join(legend_pairs)}.\n"
        if legend_pairs
        else ""
    )
    write_tex(tex + legend, Path(output_tex))
