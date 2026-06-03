"""End-to-end smoke for ``analysis/make_figures.py --smoke``.

Drives the figure generators against the hand-crafted CSV fixtures in
``tests/data/analysis_smoke/`` and asserts every generator wrote a
non-empty PDF whose magic bytes match the PDF spec.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from analysis.make_figures import main

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "analysis_smoke"

EXPECTED_PDFS = [
    "rq1_headline.pdf",
    "rq2_kr_pareto.pdf",
    "rq3_close_analogues.pdf",
    "rq4_ilp_gap.pdf",
    "rq5_persistent.pdf",
    "ablation_ordering.pdf",
    "ablation_solver.pdf",
    "ablation_budget.pdf",
]


def _assert_valid_pdf(path: Path) -> None:
    assert path.exists(), f"missing figure: {path}"
    assert path.stat().st_size > 0, f"empty figure: {path}"
    with path.open("rb") as fh:
        head = fh.read(4)
    assert head == b"%PDF", f"not a PDF (magic={head!r}): {path}"


@pytest.mark.integration
def test_make_figures_smoke_writes_every_pdf(tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    rc = main(
        [
            "--input-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    for name in EXPECTED_PDFS:
        _assert_valid_pdf(output_dir / name)


@pytest.mark.integration
def test_make_figures_smoke_handles_empty_csvs(tmp_path: Path) -> None:
    """An input directory with no CSVs must still produce placeholder PDFs."""
    empty_input = tmp_path / "empty"
    empty_input.mkdir()
    output_dir = tmp_path / "figures"
    rc = main(["--input-dir", str(empty_input), "--output-dir", str(output_dir)])
    assert rc == 0
    # Each generator must emit a placeholder PDF rather than crash.
    for name in EXPECTED_PDFS:
        _assert_valid_pdf(output_dir / name)


@pytest.mark.integration
def test_make_figures_smoke_flag_uses_bundled_fixture(tmp_path: Path) -> None:
    """``--smoke`` resolves to ``tests/data/analysis_smoke/`` by default."""
    # Mirror the fixture into tmp_path so we don't pollute results/.
    output_dir = tmp_path / "figures"
    rc = main(
        [
            "--smoke",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    for name in EXPECTED_PDFS:
        _assert_valid_pdf(output_dir / name)


@pytest.mark.integration
def test_make_figures_default_smoke_output_dir() -> None:
    """``--smoke`` without --output-dir writes to results/figures_smoke/.

    The smoke output dir is repository-relative; clean it up after the
    test to avoid leaving build artifacts in the working tree.
    """
    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "results" / "figures_smoke"
    pre_existed = default_out.exists()
    try:
        rc = main(["--smoke"])
        assert rc == 0
        for name in EXPECTED_PDFS:
            _assert_valid_pdf(default_out / name)
    finally:
        if not pre_existed and default_out.exists():
            shutil.rmtree(default_out)
