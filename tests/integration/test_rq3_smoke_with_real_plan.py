"""RQ3 smoke test against a hand-built, conflict-free plan.

Complements ``test_rq3_smoke.py``. The latter exercises the runner
bookkeeping on a pathological FakeSolver-generated plan (densely
overlapping BFS paths on the 4×4 fixture, where the probabilistic
certifier sometimes can't converge); this test pins the *happy path*
for the slack_certify_probabilistic row — when the nominal solver
returns a clean pairwise-disjoint plan, the certifier is a no-op and
the row must end at ``status="ok"`` with ``total_waits=0``.

The btpg_max and kottinger_offline rows are asserted *present*, but
not pinned to ``"ok"``: both wrappers depend on external machinery
that may or may not be available in the test environment.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml
from experiments.runners._base import CellResult
from experiments.runners.rq3_close_analogues import RQ3CloseAnaloguesRunner

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath


def _write_open_5x5_map(path: Path) -> None:
    """Write a 5×5 obstacle-free MovingAI map."""
    path.write_text(
        "type octile\nheight 5\nwidth 5\nmap\n" + "\n".join("....." for _ in range(5)) + "\n",
        encoding="utf-8",
    )


def _write_three_disjoint_agents_scen(path: Path) -> None:
    """Three agents on rows 0, 2, 4 — pairwise-disjoint cell sets."""
    rows = [
        "version 1",
        "\t".join(["0", "disjoint_5x5", "5", "5", "0", "0", "4", "0", "4"]),
        "\t".join(["1", "disjoint_5x5", "5", "5", "0", "2", "4", "2", "4"]),
        "\t".join(["2", "disjoint_5x5", "5", "5", "0", "4", "4", "4", "4"]),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _clean_three_agent_plan() -> Plan:
    """Three agents straight along rows 0, 2, 4 — zero shared cells."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(4, 0)),
        Agent(id=1, start=(0, 2), goal=(4, 2)),
        Agent(id=2, start=(0, 4), goal=(4, 4)),
    ]
    paths = [
        PlanPath(agent_id=0, vertices=[(x, 0) for x in range(5)]),
        PlanPath(agent_id=1, vertices=[(x, 2) for x in range(5)]),
        PlanPath(agent_id=2, vertices=[(x, 4) for x in range(5)]),
    ]
    return Plan.from_paths(agents, paths)


class _InjectedPlanRunner(RQ3CloseAnaloguesRunner):
    """RQ3 runner that bypasses the solver by returning a pre-built plan."""

    fixed_plan: Plan | None = None

    def _solve_nominal(
        self,
        graph: GridGraph,
        agents: list[Agent],
        cfg: dict[str, Any],
        result: CellResult,
    ) -> Plan | None:
        # No solver runtime to record — leave the default (0.0).
        result.diagnostics["solver_fallback"] = False
        result.diagnostics["nominal_injected"] = True
        return self.fixed_plan


def _smoke_config(path: Path) -> None:
    payload = {
        "experiment_name": "rq3_real_plan_smoke",
        "maps": ["disjoint_5x5"],
        "agent_counts": [3],
        "scen_seeds": [1],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "methods": [
            "slack_certify_probabilistic",
            "btpg_max",
            "kottinger_offline",
        ],
        # delta=1 aligns the certifier with the simulator's any-tick
        # overlap semantics. Disjoint paths produce zero conflicts so
        # the certifier is a no-op regardless, but epsilon=0.05 is
        # paper-grade and would catch any regression that re-introduced
        # spurious waits.
        "slack_certify_probabilistic": {"delta": 1, "p_d": 0.03, "epsilon": 0.05},
        "bernoulli": {"p_d": 0.03},
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_rq3_with_disjoint_3_agent_plan_probabilistic_ok(tmp_path: Path) -> None:
    bench = tmp_path / "bench"
    (bench / "maps").mkdir(parents=True)
    (bench / "scenarios").mkdir(parents=True)
    _write_open_5x5_map(bench / "maps" / "disjoint_5x5.map")
    _write_three_disjoint_agents_scen(bench / "scenarios" / "disjoint_5x5-random-1.scen")

    config = tmp_path / "config.yaml"
    _smoke_config(config)
    csv_path = tmp_path / "out.csv"
    manifest = tmp_path / "manifest.jsonl"

    runner = _InjectedPlanRunner(
        config_path=config,
        output_csv=csv_path,
        manifest_path=manifest,
        benchmarks_root=bench,
    )
    runner.fixed_plan = _clean_three_agent_plan()
    runner.run_all()

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # 1 map × 1 n × 1 seed × 3 methods = 3 rows.
    assert len(rows) == 3

    methods_seen = sorted(r["method"] for r in rows)
    assert methods_seen == sorted(["slack_certify_probabilistic", "btpg_max", "kottinger_offline"])

    by_method: dict[str, dict[str, str]] = {r["method"]: r for r in rows}

    # The slack_certify_probabilistic row pins the happy path: disjoint
    # paths => zero conflicts, zero waits inserted, certifier no-ops to
    # success_rate=1.0 against Bernoulli(p_d=0.03).
    prob_row = by_method["slack_certify_probabilistic"]
    assert prob_row["status"] == "ok", (
        f"unexpected status {prob_row['status']!r} for "
        f"slack_certify_probabilistic: {prob_row.get('error_message')}"
    )
    assert int(prob_row["total_waits"]) == 0
    assert int(prob_row["n_conflicts_initial"]) == 0
    sr = float(prob_row["success_rate"])
    lo = float(prob_row["success_rate_ci_lo"])
    hi = float(prob_row["success_rate_ci_hi"])
    assert 0.0 <= sr <= 1.0
    assert lo - 1e-9 <= sr <= hi + 1e-9
    assert sr == pytest.approx(1.0), (
        f"slack_certify_probabilistic success_rate={sr} should be 1.0 " f"on a fully-disjoint plan"
    )

    # btpg_max and kottinger_offline rows are present; both require
    # upstream binaries that aren't built in the sandbox, so the
    # expected sandbox outcome is ``status="binary_missing"`` (post-D3
    # for Kottinger; long-standing for btpg). The "ok" branch keeps
    # the assertion useful on machines where the binaries are built.
    allowed_baseline = {"ok", "binary_missing"}
    for method in ("btpg_max", "kottinger_offline"):
        row = by_method[method]
        assert row["status"] in allowed_baseline, (
            f"unexpected status {row['status']!r} for method "
            f"{method!r}: {row.get('error_message')}"
        )
