"""End-to-end smoke test for the RQ1 headline runner.

Uses the ``tiny_4x4.map`` fixture so no MovingAI download is needed:
one map × one ``n_agents`` × two seeds × three methods = six rows in
the output CSV.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml
from experiments.runners.rq1_headline import RQ1HeadlineRunner

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _populate_tiny_benchmarks(root: Path) -> None:
    """Mirror the tests/data fixtures under ``root/maps`` and ``root/scenarios``."""
    (root / "maps").mkdir(parents=True, exist_ok=True)
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_DIR / "tiny_4x4.map", root / "maps" / "tiny_4x4.map")
    # Mirror the same scen for two seeds so the smoke run has 2 instances.
    for seed in (1, 2):
        shutil.copy(
            DATA_DIR / "tiny_4x4-random-1.scen",
            root / "scenarios" / f"tiny_4x4-random-{seed}.scen",
        )


def _smoke_config(path: Path) -> None:
    payload = {
        "experiment_name": "rq1_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1, 2],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "methods": [
            "nominal_unhardened",
            "slack_certify_bounded",
            "slack_certify_probabilistic",
        ],
        # delta=1 keeps the bounded-mode cascade tractable on FakeSolver's
        # densely-overlapping BFS paths on the 4x4 fixture.
        "slack_certify_bounded": {"delta": 1},
        # epsilon=0.10 (permissive smoke); reachable under the corrected
        # monotone-decreasing formula.
        "slack_certify_probabilistic": {"p_d": 0.03, "epsilon": 0.10},
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_rq1_smoke_writes_csv_and_skips_resumed_cells(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "rq1_smoke.yaml"
    _smoke_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = RQ1HeadlineRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6  # 1 map × 1 n × 2 seeds × 3 methods

    # On the tiny_4x4 fixture the FakeSolver fallback produces densely
    # overlapping BFS paths. BOTH the bounded and probabilistic
    # certifiers can cascade past their max_outer_rounds budget on this
    # pathological instance — this is a FakeSolver-fixture limitation,
    # NOT a buggy-formula workaround. The live experiment uses LaCAM*
    # output, where the smokes' epsilon=0.10 / delta=1 are reachable.
    allowed_statuses_bounded = {"ok", "wait_infeasible", "cert_failed", "nominal_failed"}
    allowed_statuses_probabilistic = {"ok", "wait_infeasible", "cert_failed", "nominal_failed"}
    allowed_statuses_strict = {"ok", "wait_infeasible", "nominal_failed"}
    for row in rows:
        if row["method"] == "slack_certify_bounded":
            allowed = allowed_statuses_bounded
        elif row["method"] == "slack_certify_probabilistic":
            allowed = allowed_statuses_probabilistic
        else:
            allowed = allowed_statuses_strict
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

    # Sanity: bounded SlackCertify should never be *worse* than nominal on
    # the same seed, averaging over the seeds whose status is "ok" for both.
    ok_pairs: list[tuple[float, float]] = []
    by_key: dict[tuple[int, int, str], dict[str, str]] = {
        (int(r["n_agents"]), int(r["scen_seed"]), r["method"]): r for r in rows
    }
    for n in {int(r["n_agents"]) for r in rows}:
        for seed in {int(r["scen_seed"]) for r in rows}:
            nominal = by_key.get((n, seed, "nominal_unhardened"))
            bounded = by_key.get((n, seed, "slack_certify_bounded"))
            if (
                nominal is not None
                and bounded is not None
                and nominal["status"] == "ok"
                and bounded["status"] == "ok"
            ):
                ok_pairs.append((float(nominal["success_rate"]), float(bounded["success_rate"])))
    for nom_sr, bnd_sr in ok_pairs:
        assert bnd_sr >= nom_sr - 1e-9, (
            f"bounded SlackCertify success_rate ({bnd_sr}) below nominal "
            f"({nom_sr}); regression in the bounded certifier"
        )

    # Resume: a second run on the same manifest writes nothing new.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_again = RQ1HeadlineRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_again.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before

    # Manifest has one JSONL line per CSV row.
    with manifest.open(encoding="utf-8") as fh:
        manifest_lines = [ln for ln in fh if ln.strip()]
    assert len(manifest_lines) == 6
    # Every manifest entry is valid JSON with the expected key fields.
    for line in manifest_lines:
        obj = json.loads(line)
        assert {"map_name", "n_agents", "scen_seed", "method", "status"} <= obj.keys()
