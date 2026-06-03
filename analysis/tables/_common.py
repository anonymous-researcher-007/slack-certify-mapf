"""Shared LaTeX-formatting helpers for the table generators.

Every public formatter returns a *LaTeX-safe* string ready to drop
into a cell. Cell values that may contain arbitrary text (map names,
method names, status labels) MUST be passed through
:func:`escape_latex` first; the booktabs/tabular helpers do not do
that automatically because they often receive already-formatted
sub-expressions (math mode, percent signs, etc).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

__all__ = [
    "STATUS_ABBREVIATIONS",
    "booktabs_footer",
    "booktabs_header",
    "escape_latex",
    "format_ci",
    "format_pct",
    "format_status_counts",
    "load_csv",
    "successful_rows",
    "wrap_in_tabular",
    "write_tex",
]


STATUS_ABBREVIATIONS: dict[str, str] = {
    "ok": "OK",
    "wait_infeasible": "WI",
    "cert_failed": "CF",
    "rollout_failed": "RF",
    "nominal_failed": "NF",
    "timeout": "TO",
    "binary_missing": "BM",
}


_LATEX_SPECIAL_CHARS: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "^": r"\^{}",
    "~": r"\~{}",
    "{": r"\{",
    "}": r"\}",
}


def escape_latex(value: object) -> str:
    """Return ``value`` rendered as a LaTeX-safe string.

    Backslashes are escaped first so subsequent replacements don't
    re-process the substitutions we just inserted.

    Examples
    --------
    >>> escape_latex("warehouse-10-20_2")
    'warehouse-10-20\\\\_2'
    >>> escape_latex("100%")
    '100\\\\%'
    """
    s = "" if value is None else str(value)
    out_chars: list[str] = []
    for ch in s:
        out_chars.append(_LATEX_SPECIAL_CHARS.get(ch, ch))
    return "".join(out_chars)


def format_ci(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    """Return ``"$mean_{-down}^{+up}$"`` LaTeX math-mode string.

    Examples
    --------
    >>> format_ci(0.85, 0.82, 0.89, digits=2)
    '$0.85_{-0.03}^{+0.04}$'
    """
    if not math.isfinite(mean) or not math.isfinite(lo) or not math.isfinite(hi):
        return r"$\mathrm{n/a}$"
    down = max(0.0, mean - lo)
    up = max(0.0, hi - mean)
    return f"${mean:.{digits}f}_{{-{down:.{digits}f}}}^{{+{up:.{digits}f}}}$"


def format_pct(value: float, digits: int = 1) -> str:
    """Return ``"12.3\\%"`` — the percent sign is LaTeX-escaped.

    Examples
    --------
    >>> format_pct(0.123, digits=1)
    '12.3\\\\%'
    >>> format_pct(0.5, digits=0)
    '50\\\\%'
    """
    if not math.isfinite(value):
        return r"$\mathrm{n/a}$"
    return f"{value * 100.0:.{digits}f}\\%"


def format_status_counts(statuses: Iterable[str]) -> str:
    """Return ``"73 OK / 18 WI / 9 CF"`` for an iterable of status strings.

    Unknown statuses are reported as their literal value. Order
    follows the canonical sequence in :data:`STATUS_ABBREVIATIONS`
    (rather than insertion order) so the breakdown is stable across
    calls.

    Examples
    --------
    >>> format_status_counts(["ok", "ok", "wait_infeasible"])
    '2 OK / 1 WI'
    """
    counts: dict[str, int] = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    parts: list[str] = []
    for canonical in STATUS_ABBREVIATIONS:
        if canonical in counts:
            parts.append(f"{counts[canonical]} {STATUS_ABBREVIATIONS[canonical]}")
    # Append any unknown statuses verbatim so we never silently drop a
    # row category — that would mask a bug in the runner.
    for unknown, n in counts.items():
        if unknown not in STATUS_ABBREVIATIONS:
            parts.append(f"{n} {escape_latex(unknown)}")
    return " / ".join(parts) if parts else "--"


def booktabs_header(*column_names: str, alignment: str | None = None) -> str:
    """Return ``\\toprule\n<headers> \\\\\n\\midrule\n``.

    Column names are emitted in bold via ``\\textbf{}``; pass them
    pre-escaped if they contain LaTeX specials.

    Examples
    --------
    >>> booktabs_header("Map", "Method").splitlines()[0]
    '\\\\toprule'
    """
    _ = alignment  # kept for API compatibility; alignment is set by wrap_in_tabular
    bolded = " & ".join(f"\\textbf{{{c}}}" for c in column_names)
    return f"\\toprule\n{bolded} \\\\\n\\midrule\n"


def booktabs_footer() -> str:
    """Return ``\\bottomrule\n``."""
    return "\\bottomrule\n"


def wrap_in_tabular(body: str, n_cols: int, alignment: str | None = None) -> str:
    """Wrap ``body`` in ``\\begin{tabular}{...} ... \\end{tabular}``.

    ``alignment`` defaults to a left-then-rights pattern (``l`` for
    the first column, ``r`` for the remaining columns), which fits
    every table in this module — column 1 is a label, columns 2+ are
    numeric.

    Examples
    --------
    >>> tex = wrap_in_tabular("row1 \\\\\\\\\\n", n_cols=2)
    >>> tex.startswith("\\\\begin{tabular}{lr}")
    True
    """
    if alignment is None:
        alignment = "l" + "r" * max(0, n_cols - 1) if n_cols > 0 else "l"
    return f"\\begin{{tabular}}{{{alignment}}}\n{body}\\end{{tabular}}\n"


def write_tex(text: str, output_tex: Path) -> None:
    """Write ``text`` to ``output_tex``, creating parent directories."""
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(text, encoding="utf-8")


# ----------------------------------------------- CSV-loading convenience


def load_csv(path: Path) -> pd.DataFrame:
    """Load a Phase 7 CSV; return empty DataFrame if missing."""
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose status is ``"ok"``."""
    if df.empty or "status" not in df.columns:
        return df.iloc[0:0]
    return df[df["status"] == "ok"].copy()
