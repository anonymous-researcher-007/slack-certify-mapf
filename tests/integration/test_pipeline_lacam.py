"""Integration test for the LaCAM* wrapper.

Same skip-or-fake strategy as
:mod:`tests.integration.test_pipeline_eecbs`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent
from slackcertify.repair import slack_certify
from slackcertify.solvers._fake import FakeSolver
from slackcertify.solvers.lacam import LaCAMStarSolver

REPO_ROOT = Path(__file__).resolve().parents[2]
LACAM_BIN = REPO_ROOT / "src/slackcertify/solvers/external_bin/lacam_star"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.integration
def test_lacam_parser_on_fixture_stdout() -> None:
    fixture = (FIXTURES / "lacam_sample.txt").read_text(encoding="utf-8")
    # Real loop.map output: 3 agents, 12 timesteps (0..11).
    agents = [
        Agent(id=0, start=(0, 0), goal=(0, 2)),
        Agent(id=1, start=(0, 2), goal=(0, 1)),
        Agent(id=2, start=(0, 1), goal=(0, 3)),
    ]
    solver = LaCAMStarSolver(binary_path="/dev/null")
    plan = solver._parse_output(fixture, agents)  # noqa: SLF001 - parser test
    # 12 timesteps (0..11) -> makespan 11.
    assert plan.makespan == 11
    # Agent 0 column across timesteps 0..11.
    assert plan.paths[0].vertices == [
        (0, 0), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2),
        (0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2),
    ]
    # Agent 2 column ends at its goal (0, 3).
    assert plan.paths[2].vertices[-1] == (0, 3)


@pytest.mark.integration
def test_lacam_pipeline_uses_fake_solver_when_binary_missing() -> None:
    if shutil.which("lacam_star") is None and not LACAM_BIN.exists():
        grid = GridGraph(4, 4, set())
        agents = [
            Agent(id=0, start=(0, 0), goal=(3, 0)),
            Agent(id=1, start=(0, 3), goal=(3, 3)),
        ]
        plan = FakeSolver().solve(grid, agents, time_limit_s=0.0)
        certified, cert = slack_certify(plan, mode="bounded", delta=0)
        assert cert.verify(certified)
        return
    pytest.skip("lacam_star binary present; subprocess test deferred to Phase 5.1b")
