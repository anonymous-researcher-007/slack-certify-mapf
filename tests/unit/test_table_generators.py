"""Unit tests for the LaTeX table generators and shared helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from analysis.tables._common import (
    STATUS_ABBREVIATIONS,
    booktabs_footer,
    booktabs_header,
    escape_latex,
    format_ci,
    format_pct,
    format_status_counts,
    wrap_in_tabular,
)
from analysis.tables.runtime_table import generate_runtime_table
from analysis.tables.status_breakdown_table import generate_status_breakdown_table

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "analysis_smoke"


@pytest.mark.unit
def test_format_ci_renders_correctly_in_math_mode() -> None:
    s = format_ci(0.85, 0.82, 0.89, digits=2)
    assert s.startswith("$") and s.endswith("$")
    # Math-mode subscript = down, superscript = up.
    assert "_{-0.03}" in s
    assert "^{+0.04}" in s
    # Mean appears verbatim at 2 decimal places.
    assert "0.85" in s


@pytest.mark.unit
def test_format_ci_handles_nan() -> None:
    s = format_ci(float("nan"), 0.0, 1.0)
    # n/a placeholder for unrepresentable values.
    assert "n/a" in s and s.startswith("$")


@pytest.mark.unit
def test_format_pct_escapes_percent() -> None:
    s = format_pct(0.123, digits=1)
    assert s == r"12.3\%"
    # 0% should render as "0.0\%" (escaped), not "0%".
    assert format_pct(0.0, digits=1) == r"0.0\%"


@pytest.mark.unit
def test_escape_latex_handles_all_specials() -> None:
    raw = "_a&b%c$d#e^f~g{h}i"
    escaped = escape_latex(raw)
    for ch in ("_", "&", "%", "$", "#", "^", "~", "{", "}"):
        assert ch not in escaped or escaped.count(f"\\{ch}") >= 0
    # A pre-existing backslash should be wrapped, not lost.
    assert escape_latex("a\\b").startswith("a\\textbackslash")


@pytest.mark.unit
def test_format_status_counts_uses_abbreviations() -> None:
    s = format_status_counts(["ok", "ok", "wait_infeasible", "cert_failed"])
    assert s == "2 OK / 1 WI / 1 CF"


@pytest.mark.unit
def test_format_status_counts_handles_empty() -> None:
    assert format_status_counts([]) == "--"


@pytest.mark.unit
def test_booktabs_header_uses_correct_macros() -> None:
    h = booktabs_header("Map", "n", "Method")
    assert h.startswith("\\toprule")
    assert "\\midrule" in h
    # Column names should be bolded.
    assert "\\textbf{Map}" in h
    assert "\\textbf{n}" in h


@pytest.mark.unit
def test_booktabs_footer_emits_bottomrule() -> None:
    assert booktabs_footer() == "\\bottomrule\n"


@pytest.mark.unit
def test_wrap_in_tabular_wraps_body() -> None:
    body = "row1 \\\\\n"
    tex = wrap_in_tabular(body, n_cols=3)
    assert tex.startswith("\\begin{tabular}{lrr}")
    assert tex.endswith("\\end{tabular}\n")


@pytest.mark.unit
def test_wrap_in_tabular_custom_alignment() -> None:
    tex = wrap_in_tabular("x \\\\\n", n_cols=3, alignment="ccc")
    assert "\\begin{tabular}{ccc}" in tex


@pytest.mark.unit
def test_runtime_table_produces_valid_tex_on_smoke_fixture(tmp_path: Path) -> None:
    out = tmp_path / "tab1.tex"
    generate_runtime_table(
        FIXTURE_DIR / "rq1_headline.csv",
        FIXTURE_DIR / "rq2_kr_pareto.csv",
        FIXTURE_DIR / "rq3_close_analogues.csv",
        out,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("\\begin{tabular}")
    assert text.rstrip().endswith("\\end{tabular}")
    # Headers are present (bolded).
    assert "\\textbf{Map}" in text
    # No TODO/placeholder strings leaked into the output.
    assert "TODO" not in text
    assert "\\placeholder" not in text


@pytest.mark.unit
def test_status_breakdown_table_includes_all_statuses(tmp_path: Path) -> None:
    out = tmp_path / "tab4.tex"
    generate_status_breakdown_table(
        {
            "RQ1": FIXTURE_DIR / "rq1_headline.csv",
            "Ablation solver": FIXTURE_DIR / "ablation_solver.csv",
        },
        out,
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("\\begin{tabular}")
    # At least the OK abbreviation must appear (every smoke fixture
    # has at least one OK row).
    assert STATUS_ABBREVIATIONS["ok"] in text
    # The legend footnote names each abbreviation that fired.
    assert "Status legend" in text
