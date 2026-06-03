"""End-to-end smoke test for the RQ3 close-analogues runner.

Uses the ``tiny_4x4.map`` fixture: 1 map × 1 n × 2 seeds × 3 methods = 6
rows. The BTPG binary is not built in the test environment so we expect
``btpg_max`` rows to come back ``binary_missing``. Kottinger has an
in-process reimplementation, so its rows can land at ``"ok"`` even
without the upstream binary.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
import yaml
from experiments.runners.rq3_close_analogues import RQ3CloseAnaloguesRunner

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _populate_tiny_benchmarks(root: Path) -> None:
    (root / "maps").mkdir(parents=True, exist_ok=True)
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_DIR / "tiny_4x4.map", root / "maps" / "tiny_4x4.map")
    for seed in (1, 2):
        shutil.copy(
            DATA_DIR / "tiny_4x4-random-1.scen",
            root / "scenarios" / f"tiny_4x4-random-{seed}.scen",
        )


def _smoke_config(path: Path) -> None:
    payload = {
        "experiment_name": "rq3_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1, 2],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "methods": [
            "slack_certify_probabilistic",
            "btpg_max",
            "kottinger_offline",
        ],
        # delta=1 keeps the certifier predicate aligned with the
        # simulator's any-tick-overlap collision semantics.
        # epsilon=0.10 (permissive smoke) so the certifier converges
        # on the densely-overlapping FakeSolver BFS paths.
        "slack_certify_probabilistic": {"delta": 1, "p_d": 0.03, "epsilon": 0.10},
        "bernoulli": {"p_d": 0.03},
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_rq3_smoke_writes_csv(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "rq3_smoke.yaml"
    _smoke_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = RQ3CloseAnaloguesRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 1 map × 1 n × 2 seeds × 3 methods = 6.
    assert len(rows) == 6

    allowed_prob = {"ok", "wait_infeasible", "cert_failed", "nominal_failed"}
    # btpg_max requires the upstream binary and degrades gracefully to
    # binary_missing when it's absent (we never expect it to crash on
    # the smoke).
    allowed_btpg = {"ok", "binary_missing"}
    # Kottinger: post-D3 the wrapper raises SolverNotFoundError when
    # the upstream binary is absent (no silent reimpl fallback), so
    # the runner records status="binary_missing" in the sandbox.
    allowed_kottinger = {"ok", "binary_missing"}
    for row in rows:
        if row["method"] == "slack_certify_probabilistic":
            allowed = allowed_prob
        elif row["method"] == "btpg_max":
            allowed = allowed_btpg
        elif row["method"] == "kottinger_offline":
            allowed = allowed_kottinger
        else:
            pytest.fail(f"unexpected method {row['method']!r}")
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

    # The btpg_max wrapper must degrade cleanly (no crash) when the
    # binary is missing — the runner is responsible for catching the
    # SolverNotFoundError and reporting binary_missing.
    btpg_rows = [r for r in rows if r["method"] == "btpg_max"]
    assert btpg_rows, "expected btpg_max rows in the CSV"
    for row in btpg_rows:
        assert row["status"] in allowed_btpg

    # Resume idempotence: a second run against the same manifest must
    # append zero new CSV rows. Catches regressions where a cell's
    # manifest key drifts between runs.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_again = RQ3CloseAnaloguesRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_again.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before
