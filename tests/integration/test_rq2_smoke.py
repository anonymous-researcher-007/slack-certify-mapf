"""End-to-end smoke test for the RQ2 k-Robust Pareto runner.

Uses the ``tiny_4x4.map`` fixture so no MovingAI download is needed:
1 map × 1 ``n_agents`` × 2 seeds × 2 deltas × 3 methods = 12 rows.

The k-Robust EECBS/PBS binaries are not built in the test environment
so those methods are expected to come back as ``binary_missing``.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml
from experiments.runners.rq2_kr_pareto import RQ2KRParetoRunner

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
        "experiment_name": "rq2_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1, 2],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "delta_sweep": [0, 1],
        "methods": ["slack_certify_bounded", "kr_eecbs", "kr_pbs"],
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_rq2_smoke_writes_csv_and_skips_resumed_cells(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "rq2_smoke.yaml"
    _smoke_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = RQ2KRParetoRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 1 map × 1 n × 2 seeds × 2 deltas × 3 methods = 12.
    assert len(rows) == 12

    allowed_bounded = {"ok", "wait_infeasible", "cert_failed", "nominal_failed"}
    for row in rows:
        # Every row must carry its sweep delta in the CSV column.
        assert row["delta"] in ("0", "1"), f"unexpected delta {row['delta']!r} in row {row}"
        if row["method"] == "slack_certify_bounded":
            assert row["status"] in allowed_bounded, (
                f"unexpected status {row['status']!r} for "
                f"slack_certify_bounded: {row.get('error_message')}"
            )
            if row["status"] == "ok":
                sr = float(row["success_rate"])
                lo = float(row["success_rate_ci_lo"])
                hi = float(row["success_rate_ci_hi"])
                assert 0.0 <= sr <= 1.0
                assert lo - 1e-9 <= sr <= hi + 1e-9
        elif row["method"] in ("kr_eecbs", "kr_pbs"):
            assert row["status"] == "binary_missing", (
                f"expected binary_missing for {row['method']}, got "
                f"{row['status']!r}: {row.get('error_message')}"
            )
        else:
            pytest.fail(f"unexpected method {row['method']!r}")

    # Resume sanity: a second run on the same manifest writes nothing.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_again = RQ2KRParetoRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_again.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before

    # Manifest has one JSONL line per CSV row, and every entry carries
    # the cell's delta so resume can disambiguate sweep cells.
    with manifest.open(encoding="utf-8") as fh:
        manifest_lines = [ln for ln in fh if ln.strip()]
    assert len(manifest_lines) == 12
    seen_keys: set[tuple[str, int, int, str, int]] = set()
    for line in manifest_lines:
        obj = json.loads(line)
        assert {
            "map_name",
            "n_agents",
            "scen_seed",
            "method",
            "delta",
            "status",
        } <= obj.keys()
        seen_keys.add(
            (
                obj["map_name"],
                int(obj["n_agents"]),
                int(obj["scen_seed"]),
                obj["method"],
                int(obj["delta"]),
            )
        )
    assert len(seen_keys) == 12  # every cell-delta-method tuple is unique
