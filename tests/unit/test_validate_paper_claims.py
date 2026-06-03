"""Unit tests for scripts/validate_paper_claims.py.

Covers schema validation, the five verdict-classification paths
(Confirmed / Stronger / Weaker / Refuted / Skipped), top-level
column metric extraction, diagnostics-key metric extraction,
paired-method comparisons, every aggregation operator, and the
CLI exit-code contract (0 iff no Refuted/Weaker fires).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# scripts/ is not a package; make it importable for the tests.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import validate_paper_claims as v  # noqa: E402

# --------------------------------------------------------------- fixtures


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_yaml(path: Path, doc: dict) -> None:
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


def _simple_threshold_claim(threshold_value: float, tolerance: float) -> dict:
    return {
        "claims": [
            {
                "id": "test_claim",
                "section": "5.1",
                "paper_sentence": "Method achieves at least X.",
                "evaluation": {
                    "source_csv": "rows.csv",
                    "filter": {"status": "ok"},
                    "metric": "success_rate",
                    "aggregation": "mean",
                    "threshold": {"operator": ">=", "value": threshold_value},
                    "tolerance": tolerance,
                },
            }
        ]
    }


def _evaluate(tmp_path: Path, claims_doc: dict, csv_rows: list[dict]) -> v.EvaluationResult:
    """Convenience: write a CSV + claims YAML, evaluate the first claim."""
    csv_path = tmp_path / "rows.csv"
    _write_csv(csv_path, csv_rows)
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, claims_doc)
    claims = v.load_claims(yaml_path)
    return v.evaluate(claims[0], tmp_path)


# --------------------------------------------------------------- verdicts


@pytest.mark.unit
def test_evaluates_simple_threshold_claim_confirmed(tmp_path: Path) -> None:
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    # Observed = 0.95 → within tolerance of 0.95 → Confirmed.
    rows = [{"status": "ok", "success_rate": 0.95}]
    result = _evaluate(tmp_path, doc, rows)
    assert result.verdict == "Confirmed"
    assert result.observed == pytest.approx(0.95)


@pytest.mark.unit
def test_evaluates_simple_threshold_claim_stronger(tmp_path: Path) -> None:
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    # Observed = 0.99 → beats threshold by > tolerance → Stronger.
    rows = [{"status": "ok", "success_rate": 0.99}]
    result = _evaluate(tmp_path, doc, rows)
    assert result.verdict == "Stronger"


@pytest.mark.unit
def test_evaluates_simple_threshold_claim_weaker(tmp_path: Path) -> None:
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    # Observed = 0.85 → in direction (positive) but below by > tolerance.
    rows = [{"status": "ok", "success_rate": 0.85}]
    result = _evaluate(tmp_path, doc, rows)
    assert result.verdict == "Weaker"


@pytest.mark.unit
def test_evaluates_simple_threshold_claim_refuted(tmp_path: Path) -> None:
    # Paired-difference style threshold: ">= 0.20" but observed
    # is negative → contradicts direction → Refuted.
    doc = {
        "claims": [
            {
                "id": "rq1_diff",
                "section": "5.1",
                "paper_sentence": "A − B ≥ 20 pp.",
                "evaluation": {
                    "source_csv": "rows.csv",
                    "paired_comparison": {
                        "filter_common": {"status": "ok", "n_agents": 20},
                        "method_a": {"method": "a"},
                        "method_b": {"method": "b"},
                        "metric": "success_rate",
                        "operation": "subtract",
                        "aggregation": "mean",
                    },
                    "threshold": {"operator": ">=", "value": 0.20},
                    "tolerance": 0.05,
                },
            }
        ]
    }
    rows = [
        {"map_name": "m", "n_agents": 20, "scen_seed": 1, "method": "a",
         "status": "ok", "success_rate": 0.40},
        {"map_name": "m", "n_agents": 20, "scen_seed": 1, "method": "b",
         "status": "ok", "success_rate": 0.55},
    ]
    csv_path = tmp_path / "rows.csv"
    _write_csv(csv_path, rows)
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    claims = v.load_claims(yaml_path)
    result = v.evaluate(claims[0], tmp_path)
    # diff = -0.15, threshold direction was >= 0.20 (positive); -0.15
    # has opposite sign → Refuted.
    assert result.verdict == "Refuted"
    assert result.observed == pytest.approx(-0.15)


# --------------------------------------------------------------- skip paths


@pytest.mark.unit
def test_skips_when_source_csv_missing(tmp_path: Path) -> None:
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    # Do NOT write the CSV — yet point at a tmp_path that doesn't
    # contain it.
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    claims = v.load_claims(yaml_path)
    result = v.evaluate(claims[0], tmp_path)
    assert result.verdict == "Skipped"
    assert result.reason.startswith("missing_csv:")


@pytest.mark.unit
def test_skips_when_no_rows_matched(tmp_path: Path) -> None:
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    # CSV exists but the filter selects zero rows.
    rows = [{"status": "cert_failed", "success_rate": 0.40}]
    result = _evaluate(tmp_path, doc, rows)
    assert result.verdict == "Skipped"
    assert result.reason == "no_rows_matched"


# ---------------------------------------------------- paired + diagnostics


@pytest.mark.unit
def test_paired_comparison_subtract(tmp_path: Path) -> None:
    """Paired diff over two methods at two seeds = mean(a_seed - b_seed)."""
    doc = {
        "claims": [
            {
                "id": "paired",
                "section": "5.1",
                "paper_sentence": "A beats B by at least 0.15.",
                "evaluation": {
                    "source_csv": "rows.csv",
                    "paired_comparison": {
                        "filter_common": {"status": "ok"},
                        "method_a": {"method": "a"},
                        "method_b": {"method": "b"},
                        "metric": "success_rate",
                        "operation": "subtract",
                        "aggregation": "mean",
                    },
                    "threshold": {"operator": ">=", "value": 0.15},
                    "tolerance": 0.02,
                },
            }
        ]
    }
    # Per-seed diffs: 0.20, 0.10 → mean 0.15 → exactly threshold → Confirmed.
    rows = [
        {"map_name": "m", "n_agents": 5, "scen_seed": 1, "method": "a",
         "status": "ok", "success_rate": 0.90},
        {"map_name": "m", "n_agents": 5, "scen_seed": 1, "method": "b",
         "status": "ok", "success_rate": 0.70},
        {"map_name": "m", "n_agents": 5, "scen_seed": 2, "method": "a",
         "status": "ok", "success_rate": 0.85},
        {"map_name": "m", "n_agents": 5, "scen_seed": 2, "method": "b",
         "status": "ok", "success_rate": 0.75},
    ]
    csv_path = tmp_path / "rows.csv"
    _write_csv(csv_path, rows)
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    claims = v.load_claims(yaml_path)
    result = v.evaluate(claims[0], tmp_path)
    assert result.observed == pytest.approx(0.15)
    assert result.verdict == "Confirmed"


@pytest.mark.unit
def test_metric_from_diagnostics(tmp_path: Path) -> None:
    """Extract a key from the JSON-encoded diagnostics column."""
    doc = {
        "claims": [
            {
                "id": "gap_max",
                "section": "5.4",
                "paper_sentence": "Max gap ≤ 25%.",
                "evaluation": {
                    "source_csv": "rows.csv",
                    "filter": {"status": "ok"},
                    "metric_from_diagnostics": "gap_pct",
                    "aggregation": "max",
                    "threshold": {"operator": "<=", "value": 25.0},
                    "tolerance": 2.0,
                },
            }
        ]
    }
    # max gap = 24.0, threshold ≤ 25.0, tolerance 2.0 → within
    # |observed − threshold| ≤ tolerance band → Confirmed.
    rows = [
        {"status": "ok",
         "diagnostics": json.dumps({"gap_pct": 0.0, "other": "x"})},
        {"status": "ok",
         "diagnostics": json.dumps({"gap_pct": 24.0})},
        {"status": "ok",
         "diagnostics": json.dumps({"gap_pct": 12.3})},
    ]
    result = _evaluate(tmp_path, doc, rows)
    assert result.observed == pytest.approx(24.0)
    assert result.verdict == "Confirmed"


# ---------------------------------------------------------- aggregations


@pytest.mark.unit
def test_aggregation_fraction_true(tmp_path: Path) -> None:
    """fraction_true counts how many bool-coerced metric values are True."""
    doc = {
        "claims": [
            {
                "id": "matched_optimum",
                "section": "5.4",
                "paper_sentence": "Greedy matches optimum on ≥ 50%.",
                "evaluation": {
                    "source_csv": "rows.csv",
                    "filter": {"status": "ok"},
                    "metric_from_diagnostics": "matched_optimum",
                    "aggregation": "fraction_true",
                    "threshold": {"operator": ">=", "value": 0.50},
                    "tolerance": 0.05,
                },
            }
        ]
    }
    rows = [
        {"status": "ok",
         "diagnostics": json.dumps({"matched_optimum": True})},
        {"status": "ok",
         "diagnostics": json.dumps({"matched_optimum": True})},
        {"status": "ok",
         "diagnostics": json.dumps({"matched_optimum": False})},
        {"status": "ok",
         "diagnostics": json.dumps({"matched_optimum": True})},
    ]
    result = _evaluate(tmp_path, doc, rows)
    # 3/4 True → 0.75.
    assert result.observed == pytest.approx(0.75)
    assert result.verdict == "Stronger"


@pytest.mark.unit
def test_aggregation_max(tmp_path: Path) -> None:
    """max aggregates by taking the largest value in the metric series."""
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    doc["claims"][0]["evaluation"]["aggregation"] = "max"
    rows = [
        {"status": "ok", "success_rate": 0.80},
        {"status": "ok", "success_rate": 0.95},
        {"status": "ok", "success_rate": 0.70},
    ]
    result = _evaluate(tmp_path, doc, rows)
    assert result.observed == pytest.approx(0.95)
    assert result.verdict == "Confirmed"


# -------------------------------------------------------- schema validation


@pytest.mark.unit
def test_yaml_schema_validation_rejects_missing_fields(tmp_path: Path) -> None:
    doc = {
        "claims": [
            {
                "id": "bad",
                "section": "5.1",
                "paper_sentence": "x",
                "evaluation": {
                    "source_csv": "rows.csv",
                    "filter": {"status": "ok"},
                    "metric": "success_rate",
                    "aggregation": "mean",
                    "threshold": {"value": 0.5},  # operator missing
                    "tolerance": 0.0,
                },
            }
        ]
    }
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    with pytest.raises(v.ClaimSchemaError, match="operator"):
        v.load_claims(yaml_path)


@pytest.mark.unit
def test_yaml_schema_validation_rejects_unknown_aggregation(tmp_path: Path) -> None:
    doc = _simple_threshold_claim(0.95, tolerance=0.0)
    doc["claims"][0]["evaluation"]["aggregation"] = "weighted_geomean"
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    with pytest.raises(v.ClaimSchemaError) as excinfo:
        v.load_claims(yaml_path)
    msg = str(excinfo.value)
    assert "weighted_geomean" in msg
    # Error must mention the supported aggregations explicitly so
    # the reader can fix the typo without grepping the source.
    for known in ("mean", "max", "min", "fraction_true"):
        assert known in msg


# ------------------------------------------------------------ CLI exit code


@pytest.mark.unit
def test_exit_code_zero_on_all_confirmed(tmp_path: Path) -> None:
    """When every claim Confirms/Stronger/Skipped, exit code is 0."""
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    rows = [{"status": "ok", "success_rate": 0.95}]
    csv_path = tmp_path / "rows.csv"
    _write_csv(csv_path, rows)
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    out = tmp_path / "report.md"
    rc = v.main(
        [
            "--results-dir", str(tmp_path),
            "--claims", str(yaml_path),
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


@pytest.mark.unit
def test_exit_code_one_on_any_refuted(tmp_path: Path) -> None:
    """When any claim Weaker or Refuted, exit code is 1."""
    doc = _simple_threshold_claim(0.95, tolerance=0.02)
    rows = [{"status": "ok", "success_rate": 0.40}]  # Weaker
    csv_path = tmp_path / "rows.csv"
    _write_csv(csv_path, rows)
    yaml_path = tmp_path / "claims.yaml"
    _write_yaml(yaml_path, doc)
    out = tmp_path / "report.md"
    rc = v.main(
        [
            "--results-dir", str(tmp_path),
            "--claims", str(yaml_path),
            "--out", str(out),
        ]
    )
    assert rc == 1
