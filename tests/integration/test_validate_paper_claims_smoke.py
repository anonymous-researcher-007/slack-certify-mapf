"""End-to-end smoke for scripts/validate_paper_claims.py.

Runs the validator against the bundled smoke claims YAML
(``tests/data/claims_smoke.yaml``) and the bundled Phase 8.1
fixture CSVs (``tests/data/analysis_smoke/``). Asserts the report
is well-formed: starts with the canonical heading, contains the
Summary section, and contains all five verdict subsections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.validate_paper_claims import main  # noqa: E402

_FIXTURE_DIR = _REPO_ROOT / "tests" / "data" / "analysis_smoke"
_CLAIMS = _REPO_ROOT / "tests" / "data" / "claims_smoke.yaml"


@pytest.mark.integration
def test_validator_runs_against_smoke_fixtures(tmp_path: Path) -> None:
    out = tmp_path / "claim_validation_smoke.md"
    tables = tmp_path / "claim_validation_smoke_tables.tex"
    rc = main(
        [
            "--results-dir", str(_FIXTURE_DIR),
            "--claims", str(_CLAIMS),
            "--out", str(out),
            "--tables-out", str(tables),
        ]
    )
    # Smoke claims are calibrated to land Confirmed / Stronger on the
    # bundled fixtures — neither blocking. Exit code should be 0.
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # Header + sections required by the spec.
    assert text.startswith("# Paper Claim Validation Report")
    assert "## Summary" in text
    for section in ("## Refuted", "## Weaker", "## Stronger",
                    "## Confirmed", "## Skipped"):
        assert section in text, f"missing section: {section!r}"
    # Every smoke claim id must appear somewhere in the report.
    for cid in (
        "smoke_rq1_certified_warehouse_n20",
        "smoke_rq4_optimality_gap_max",
        "smoke_rq1_certified_beats_nominal",
    ):
        assert cid in text, f"missing claim id in report: {cid}"
    # Tables companion exists and looks LaTeX-ish.
    assert tables.exists()
    tables_text = tables.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in tables_text
    assert "\\toprule" in tables_text
