"""RQ4 smoke test against a hand-built, conflict-free plan.

Complements ``test_rq4_smoke.py``. The latter exercises the runner
bookkeeping on the FakeSolver's densely overlapping BFS paths (where
both methods may legitimately bail at ``wait_infeasible``); this test
pins the happy path — when the nominal solver returns a clean
pairwise-disjoint plan, both methods must reach ``status="ok"`` with
zero waits inserted, and ``matched_optimum`` must hold trivially.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from experiments.runners._base import CellResult
from experiments.runners.rq4_ilp_gap import RQ4ILPGapRunner

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath


def _write_open_5x5_map(path: Path) -> None:
    path.write_text(
        "type octile\nheight 5\nwidth 5\nmap\n" + "\n".join("....." for _ in range(5)) + "\n",
        encoding="utf-8",
    )


def _write_three_disjoint_agents_scen(path: Path) -> None:
    rows = [
        "version 1",
        "\t".join(["0", "disjoint_5x5", "5", "5", "0", "0", "4", "0", "4"]),
        "\t".join(["1", "disjoint_5x5", "5", "5", "0", "2", "4", "2", "4"]),
        "\t".join(["2", "disjoint_5x5", "5", "5", "0", "4", "4", "4", "4"]),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _clean_three_agent_plan() -> Plan:
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


class _InjectedPlanRunner(RQ4ILPGapRunner):
    """RQ4 runner that bypasses the solver by returning a pre-built plan."""

    fixed_plan: Plan | None = None

    def _solve_nominal(
        self,
        graph: GridGraph,
        agents: list[Agent],
        cfg: dict[str, Any],
        result: CellResult,
    ) -> Plan | None:
        result.diagnostics["solver_fallback"] = False
        result.diagnostics["nominal_injected"] = True
        return self.fixed_plan


def _smoke_config(path: Path) -> None:
    payload = {
        "experiment_name": "rq4_real_plan_smoke",
        "maps": ["disjoint_5x5"],
        "agent_counts": [3],
        "scen_seeds": [1],
        "nominal_solver": "lacam_star",
        "nominal_time_limit_s": 1.0,
        "delta": 2,
        "methods": ["slack_certify_bounded", "ilp_exact"],
        "ilp_time_limit_s": 30.0,
        "rollouts": 50,
        "rollout_parallel_jobs": 1,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_rq4_with_disjoint_3_agent_plan_matched_optimum(tmp_path: Path) -> None:
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

    # 1 map × 1 n × 1 seed × 2 methods = 2 rows.
    assert len(rows) == 2
    by_method: dict[str, dict[str, str]] = {r["method"]: r for r in rows}
    assert set(by_method) == {"slack_certify_bounded", "ilp_exact"}

    # Happy-path: both methods land at status="ok" with zero waits.
    for method, row in by_method.items():
        assert row["status"] == "ok", (
            f"unexpected status {row['status']!r} for method {method!r}: "
            f"{row.get('error_message')}"
        )
        assert int(row["total_waits"]) == 0, (
            f"{method}: total_waits={row['total_waits']} should be 0 " f"on a fully-disjoint plan"
        )
        assert int(row["n_conflicts_initial"]) == 0
        assert float(row["success_rate"]) == pytest.approx(1.0)

    # Trivial 3-agent disjoint instance: the ILP proves optimum at 0
    # waits, so the gap diagnostics on the slack_certify_bounded row
    # must use the "to_optimum" label with ilp_proven_optimal=True
    # and matched_optimum=True (greedy trivially matches at 0).
    sb_diag = json.loads(by_method["slack_certify_bounded"]["diagnostics"])
    assert sb_diag.get("gap_label") == "to_optimum", f"gap_label missing or wrong: {sb_diag}"
    assert (
        sb_diag.get("ilp_proven_optimal") is True
    ), f"ilp_proven_optimal missing or False: {sb_diag}"
    assert sb_diag.get("matched_optimum") is True, f"matched_optimum missing or not True: {sb_diag}"
    assert sb_diag.get("ilp_waits") == 0
    assert float(sb_diag.get("gap_pct_to_optimum", float("inf"))) == pytest.approx(0.0)
