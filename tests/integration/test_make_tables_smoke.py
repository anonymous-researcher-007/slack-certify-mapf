"""End-to-end smoke for ``analysis/make_tables.py --smoke``.

Drives the table generators against the hand-crafted CSV fixtures in
``tests/data/analysis_smoke/`` and asserts every generator wrote a
non-empty ``.tex`` fragment that begins with ``\\begin{tabular}``
and contains no leftover ``TODO`` or ``\\placeholder`` markers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from analysis.make_tables import main

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "analysis_smoke"

EXPECTED_TEX = [
    "tab1_runtime.tex",
    "tab2_analogues.tex",
    "tab3_optimality_gap.tex",
    "tab4_status_breakdown.tex",
]


def _assert_valid_tex(path: Path) -> None:
    assert path.exists(), f"missing tex: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"empty tex: {path}"
    assert "\\begin{tabular}" in text, f"no tabular env in {path}: {text[:200]!r}"
    assert "\\end{tabular}" in text, f"unclosed tabular in {path}: {text[-200:]!r}"
    # Catch incomplete generators that left stub markers in.
    assert "TODO" not in text, f"TODO leaked into {path}: {text!r}"
    assert "\\placeholder" not in text, f"placeholder leaked into {path}"


@pytest.mark.integration
def test_make_tables_smoke_writes_every_tex(tmp_path: Path) -> None:
    output_dir = tmp_path / "summary"
    rc = main(
        [
            "--input-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    for name in EXPECTED_TEX:
        _assert_valid_tex(output_dir / name)


@pytest.mark.integration
def test_make_tables_smoke_handles_empty_csvs(tmp_path: Path) -> None:
    """Empty input directory must still produce a tabular per table."""
    empty_input = tmp_path / "empty"
    empty_input.mkdir()
    output_dir = tmp_path / "summary"
    rc = main(["--input-dir", str(empty_input), "--output-dir", str(output_dir)])
    assert rc == 0
    for name in EXPECTED_TEX:
        _assert_valid_tex(output_dir / name)


@pytest.mark.integration
def test_make_tables_smoke_flag_uses_bundled_fixture(tmp_path: Path) -> None:
    """``--smoke`` resolves to ``tests/data/analysis_smoke/`` by default."""
    output_dir = tmp_path / "summary"
    rc = main(["--smoke", "--output-dir", str(output_dir)])
    assert rc == 0
    for name in EXPECTED_TEX:
        _assert_valid_tex(output_dir / name)
