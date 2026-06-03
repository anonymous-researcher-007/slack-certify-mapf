"""Integration test for the pR-CBS wrapper (parser only — real-binary
test is impractical given the ~8-agent scaling limit)."""

from __future__ import annotations

from pathlib import Path

import pytest
from baselines.pr_cbs.wrapper import PRobustCBSBaseline

from slackcertify.core.plan import Agent

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.integration
def test_pr_cbs_parser_on_fixture_stdout() -> None:
    # pR-CBS reuses the EECBS plain-text path format, so the kR-EECBS
    # fixture is sufficient to exercise the parser.
    fixture = (FIXTURES / "kr_eecbs_sample.txt").read_text(encoding="utf-8")
    agents = [
        Agent(id=0, start=(0, 0), goal=(3, 0)),
        Agent(id=1, start=(3, 1), goal=(0, 1)),
    ]
    wrapper = PRobustCBSBaseline(binary_path="/dev/null")
    plan = wrapper._parse_output(fixture, agents)  # noqa: SLF001 - parser test
    assert plan.makespan == 3
    assert {p.agent_id for p in plan.paths} == {0, 1}
