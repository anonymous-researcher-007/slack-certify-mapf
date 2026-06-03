"""Integration tests for the kR-CBS / kR-PBS wrappers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from baselines.kr_cbs.wrapper import KRobustCBSBaseline

from slackcertify.certify.bounded import is_delta_certified
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent
from slackcertify.repair import slack_certify
from slackcertify.solvers._fake import FakeSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
KR_BIN = REPO_ROOT / "src/slackcertify/solvers/external_bin/kr_eecbs"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.integration
def test_kr_cbs_parser_on_fixture_stdout() -> None:
    fixture = (FIXTURES / "kr_eecbs_sample.txt").read_text(encoding="utf-8")
    agents = [
        Agent(id=0, start=(0, 0), goal=(3, 0)),
        Agent(id=1, start=(3, 1), goal=(0, 1)),
    ]
    wrapper = KRobustCBSBaseline(binary_path="/dev/null")
    plan = wrapper._parse_output(fixture, agents)  # noqa: SLF001 - parser test
    assert plan.makespan == 3
    assert plan.paths[0].vertices[0] == (0, 0)
    assert plan.paths[1].vertices[-1] == (0, 1)


@pytest.mark.integration
def test_kr_cbs_pipeline_uses_fake_solver_when_binary_missing() -> None:
    if shutil.which("kr_eecbs") is not None or KR_BIN.exists():
        pytest.skip("kr_eecbs binary present; subprocess test deferred to Phase 5.1b")

    grid = GridGraph(4, 4, set())
    agents = [
        Agent(id=0, start=(1, 0), goal=(3, 0)),
        Agent(id=1, start=(2, 3), goal=(2, 0)),
    ]
    plan = FakeSolver().solve(grid, agents, time_limit_s=0.0)
    # Apply Slack-Certify with delta=2 to enforce Δ-disjointness on the
    # comparison axis. The wrapper itself is exercised in the parser
    # test above; this branch only sanity-checks the surrounding
    # pipeline.
    certified, _ = slack_certify(plan, mode="bounded", delta=2)
    assert is_delta_certified(certified, delta=2)
