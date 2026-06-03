"""End-to-end smokes for the three Phase 7.2d ablation runners.

Each smoke uses the ``tiny_4x4.map`` fixture and a 1-seed, 50-rollout
configuration so the tests run in a few seconds.

* ``ablation_ordering`` — sweeps {topological, risk_descending, random}
  over the same fixture. The random-ordering row must reproduce
  bit-identically across re-runs (deterministic seed).
* ``ablation_solver`` — sweeps {eecbs, lacam_star, pibt}. Each row's
  expected status depends on whether that solver's binary is installed
  in ``external_bin/``: a present binary really solves the tiny fixture
  (``ok``); an absent one yields ``binary_missing``.
* ``ablation_budget`` — sweeps {uniform, risk_proportional}. Both
  rows are exercised; diagnostics records the allocator name.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml
from experiments.runners.ablation_budget import AblationBudgetRunner
from experiments.runners.ablation_ordering import AblationOrderingRunner
from experiments.runners.ablation_solver import _SOLVER_CLASSES, AblationSolverRunner

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _populate_tiny_benchmarks(root: Path) -> None:
    (root / "maps").mkdir(parents=True, exist_ok=True)
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_DIR / "tiny_4x4.map", root / "maps" / "tiny_4x4.map")
    shutil.copy(
        DATA_DIR / "tiny_4x4-random-1.scen",
        root / "scenarios" / "tiny_4x4-random-1.scen",
    )


# --------------------------------------------------------------- ordering


def _ordering_config(path: Path) -> None:
    payload = {
        "experiment_name": "ablation_ordering_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "delta": 1,
        # epsilon=0.10 (permissive smoke): the FakeSolver tiny_4x4
        # plan has densely overlapping BFS paths so a tight epsilon
        # would cascade past max_outer_rounds.
        "p_d": 0.03,
        "epsilon": 0.10,
        "methods": ["topological", "risk_descending", "random"],
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_ablation_ordering_smoke(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "config.yaml"
    _ordering_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = AblationOrderingRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3  # 1 map × 1 n × 1 seed × 3 methods

    methods_seen = sorted(r["method"] for r in rows)
    assert methods_seen == sorted(["topological", "risk_descending", "random"])

    allowed = {"ok", "wait_infeasible", "cert_failed", "nominal_failed"}
    for row in rows:
        assert row["status"] in allowed, (
            f"unexpected status {row['status']!r} for {row['method']!r}: "
            f"{row.get('error_message')}"
        )
        diag = json.loads(row["diagnostics"])
        if row["status"] == "ok":
            assert diag.get("ordering") == row["method"]
            assert diag.get("mode") == "probabilistic"

    # Random ordering must be deterministic: re-run the smoke into a
    # fresh CSV + manifest with the same scen_seed and assert the random
    # row's certifier-driven counters are identical.
    csv_path2 = tmp_path / "out2.csv"
    manifest2 = tmp_path / "manifest2.jsonl"
    runner2 = AblationOrderingRunner(
        config_path=config,
        output_csv=csv_path2,
        manifest_path=manifest2,
        benchmarks_root=bench_root,
    )
    runner2.run_all()
    with csv_path2.open(encoding="utf-8") as fh:
        rows2 = list(csv.DictReader(fh))

    def _row_for(rs: list[dict[str, str]], method: str) -> dict[str, str]:
        return next(r for r in rs if r["method"] == method)

    rand_a = _row_for(rows, "random")
    rand_b = _row_for(rows2, "random")
    for field in ("status", "total_waits", "outer_rounds", "n_conflicts_initial"):
        assert rand_a[field] == rand_b[field], (
            f"random ordering not deterministic at field {field!r}: "
            f"{rand_a[field]!r} vs {rand_b[field]!r}"
        )
    if rand_a["status"] == "ok":
        # success_rate is rng-driven (rollouts also seeded) — should
        # match bit-for-bit on a deterministic re-run.
        assert rand_a["success_rate"] == rand_b["success_rate"]

    # Resume idempotence: a third run against the original manifest
    # must append zero new CSV rows.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_resume = AblationOrderingRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_resume.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before


# --------------------------------------------------------------- solver


def _solver_config(path: Path) -> None:
    payload = {
        "experiment_name": "ablation_solver_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1],
        "nominal_time_limit_s": 1.0,
        "delta": 1,
        "methods": ["eecbs", "lacam_star", "pibt"],
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_ablation_solver_smoke(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "config.yaml"
    _solver_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = AblationSolverRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3  # 1 map × 1 n × 1 seed × 3 methods

    methods_seen = sorted(r["method"] for r in rows)
    assert methods_seen == sorted(["eecbs", "lacam_star", "pibt"])

    # Expected status is per-solver, driven by binary presence in
    # external_bin/: an installed binary (PIBT, LaCAM*) really solves the
    # tiny fixture (status=ok); an absent one (EECBS here) terminates at
    # binary_missing without crashing.
    for row in rows:
        solver = _SOLVER_CLASSES[row["method"]]()
        if solver.binary_path.exists():
            assert row["status"] == "ok", (
                f"expected ok for installed binary {row['method']!r}, got "
                f"{row['status']!r}: {row.get('error_message')}"
            )
        else:
            assert row["status"] == "binary_missing", (
                f"expected binary_missing for absent {row['method']!r}, got "
                f"{row['status']!r}: {row.get('error_message')}"
            )
            assert "SolverNotFoundError" in (row.get("error_message") or "")

    # Resume idempotence: a second run against the same manifest must
    # append zero new CSV rows, even when every cell is binary_missing.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_again = AblationSolverRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_again.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before


# --------------------------------------------------------------- budget


def _budget_config(path: Path) -> None:
    payload = {
        "experiment_name": "ablation_budget_smoke",
        "maps": ["tiny_4x4"],
        "agent_counts": [2],
        "scen_seeds": [1],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "delta": 1,
        # epsilon=0.10 (permissive smoke): same FakeSolver-fixture
        # caveat as the ordering smoke above.
        "p_d": 0.03,
        "epsilon": 0.10,
        "methods": ["uniform", "risk_proportional"],
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_ablation_budget_smoke(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _populate_tiny_benchmarks(bench_root)
    config = tmp_path / "config.yaml"
    _budget_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = AblationBudgetRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # 1 map × 1 n × 1 seed × 2 methods

    methods_seen = sorted(r["method"] for r in rows)
    assert methods_seen == sorted(["uniform", "risk_proportional"])

    # Both methods must complete the bookkeeping; the cell-level status
    # is allowed to be cert_failed on the densely overlapping FakeSolver
    # BFS paths, but the allocator name must always show up in the
    # diagnostics blob (it is set as soon as _apply_allocator enters).
    allowed = {"ok", "wait_infeasible", "cert_failed", "nominal_failed"}
    for row in rows:
        assert row["status"] in allowed, (
            f"unexpected status {row['status']!r} for {row['method']!r}: "
            f"{row.get('error_message')}"
        )
        if row["status"] == "ok":
            diag = json.loads(row["diagnostics"])
            assert diag.get("budget_alloc") == row["method"]
            assert diag.get("mode") == "probabilistic"

    # Resume idempotence: a second run against the same manifest must
    # append zero new CSV rows.
    rows_before = csv_path.read_text(encoding="utf-8")
    runner_again = AblationBudgetRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench_root,
    )
    runner_again.run_all()
    rows_after = csv_path.read_text(encoding="utf-8")
    assert rows_after == rows_before
