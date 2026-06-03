"""End-to-end smoke test for the RQ4 optimality-gap runner.

Uses the ``tiny_4x4.map`` fixture: 1 map × 1 ``n_agents`` × 1 seed × 2
methods = 2 rows. The ILP runs with a 30-second wall-clock budget which
is overkill for a 2-agent fixture; timeout in the smoke would indicate
a runaway encoding bug.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml
from experiments.runners.rq4_ilp_gap import RQ4ILPGapRunner

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _populate_tiny_benchmarks(root: Path) -> None:
    (root / "maps").mkdir(parents=True, exist_ok=True)
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_DIR / "tiny_4x4.map", root / "maps" / "tiny_4x4.map")
    shutil.copy(
        DATA_DIR / "tiny_4x4-random-1.scen",
        root / "scenarios" / "tiny_4x4-random-1.scen",
    )


def _smoke_config(path: Path) -> None:
    payload = {
        "experiment_name": "rq4_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        # delta=1 keeps the wait cascade tractable on FakeSolver's densely
        # overlapping BFS paths on the 4x4 fixture.
        "delta": 1,
        "methods": ["slack_certify_bounded", "ilp_exact"],
        "ilp_time_limit_s": 30.0,
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_rq4_smoke_writes_csv_with_gap_diagnostics(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "rq4_smoke.yaml"
    _smoke_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = RQ4ILPGapRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # 1 cell × 2 methods

    methods_seen = sorted(r["method"] for r in rows)
    assert methods_seen == sorted(["slack_certify_bounded", "ilp_exact"])

    allowed = {"ok", "wait_infeasible", "cert_failed", "timeout"}
    by_method: dict[str, dict[str, str]] = {r["method"]: r for r in rows}

    for row in rows:
        assert row["status"] in allowed, (
            f"unexpected status {row['status']!r} for method "
            f"{row['method']!r}: {row.get('error_message')}"
        )
        if row["status"] == "ok":
            sr = float(row["success_rate"])
            lo = float(row["success_rate_ci_lo"])
            hi = float(row["success_rate_ci_hi"])
            assert 0.0 <= sr <= 1.0
            assert lo - 1e-9 <= sr <= hi + 1e-9

    sb_row = by_method["slack_certify_bounded"]
    ilp_row = by_method["ilp_exact"]

    # The ILP row always records its wall-clock runtime in the
    # diagnostics blob, regardless of terminal status (optimal,
    # timeout, infeasible, etc.).
    ilp_diag = json.loads(ilp_row["diagnostics"])
    assert (
        "ilp_runtime_s" in ilp_diag
    ), f"missing ilp_runtime_s in ilp_exact diagnostics: {ilp_diag}"
    assert isinstance(ilp_diag["ilp_runtime_s"], (int, float))
    assert float(ilp_diag["ilp_runtime_s"]) >= 0.0

    # Cross-method gap_pct only attaches when both rows reached "ok".
    # The label distinguishes proven-optimal ILP from feasible-only:
    # gap_pct_to_optimum vs gap_pct_to_feasible. On a 2-agent fixture
    # with a 30 s budget we expect proven optimality, but the test
    # accepts either label.
    if sb_row["status"] == "ok" and ilp_row["status"] == "ok":
        sb_diag = json.loads(sb_row["diagnostics"])
        assert "gap_label" in sb_diag, f"both methods ok but no gap_label: {sb_diag}"
        label = sb_diag["gap_label"]
        assert label in (
            "to_optimum",
            "to_feasible_upper_bound",
        ), f"unexpected gap_label {label!r}: {sb_diag}"
        if label == "to_optimum":
            gap_key = "gap_pct_to_optimum"
            assert sb_diag.get("ilp_proven_optimal") is True
            assert "matched_optimum" in sb_diag
            assert isinstance(sb_diag["matched_optimum"], bool)
        else:
            gap_key = "gap_pct_to_feasible"
            assert sb_diag.get("ilp_proven_optimal") is False
            assert sb_diag.get("matched_optimum") is None
            assert "note" in sb_diag
        assert gap_key in sb_diag, f"gap_label={label!r} but {gap_key!r} missing: {sb_diag}"
        assert isinstance(sb_diag[gap_key], (int, float))
        # Greedy may match the ILP wait count but cannot drop below a
        # proven optimum; against a feasible upper bound, dropping
        # below is informative but not a bug.
        if label == "to_optimum":
            assert float(sb_diag[gap_key]) >= -1e-9, (
                f"{gap_key}={sb_diag[gap_key]} negative against proven "
                f"optimum — greedy claims fewer waits than ILP optimum"
            )
        assert "ilp_waits" in sb_diag
        assert isinstance(sb_diag["ilp_waits"], int)
        assert sb_diag["ilp_waits"] >= 0

    # Wait-infeasible consistency: if EITHER side declares
    # wait_infeasible, the other should too (both prechecks should
    # agree on the corridor-swap case). The runner attaches a
    # precheck_disagreement diagnostic when they don't.
    if sb_row["status"] == "wait_infeasible" or ilp_row["status"] == "wait_infeasible":
        assert sb_row["status"] == ilp_row["status"], (
            f"precheck disagreement: slack_certify_bounded="
            f"{sb_row['status']!r}, ilp_exact={ilp_row['status']!r}"
        )

    # Resume idempotence: a second run against the same manifest must
    # append zero new CSV rows. RQ4 is the most exposed runner here
    # because it OVERRIDES run_all to group both methods per (map, n,
    # scen_seed) and emit them together; any regression where the
    # grouped dispatch re-emits an already-completed cell would land
    # here.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_again = RQ4ILPGapRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_again.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before
