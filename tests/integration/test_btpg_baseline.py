"""Integration tests for the BTPG-max wrapper.

The BTPG binary is not built in this environment; the tests therefore
mock :func:`subprocess.run` to copy a fixture JSON to the wrapper's
``--output`` path, exercising the full read-back-and-tag flow.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from baselines.btpg.wrapper import BTPGMaxBaseline

from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_subprocess_runner(fixture_payload: str) -> object:
    """Return a callable matching ``subprocess.run`` that writes ``fixture_payload``
    to whichever path follows ``--output`` in argv."""

    def _runner(args: list[str], **kwargs: object) -> SimpleNamespace:
        _ = kwargs
        out_idx = args.index("--output")
        out_path = Path(args[out_idx + 1])
        out_path.write_text(fixture_payload, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return _runner


@pytest.mark.integration
def test_btpg_round_trip_with_mocked_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use any executable path so _check_binary_exists passes.
    sleep_path = shutil.which("true") or shutil.which("sleep")
    if sleep_path is None:
        pytest.skip("no executable available to satisfy _check_binary_exists")

    plan = Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(2, 0))],
        paths=[PlanPath(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])],
    )
    fixture_payload = (FIXTURES / "btpg_sample.json").read_text(encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_runner(fixture_payload))

    wrapper = BTPGMaxBaseline(binary_path=sleep_path)
    result = wrapper.post_process(plan)
    # The fixture round-trips into a valid Plan with two agents.
    assert len(result.plan.agents) == 2
    assert result.plan.makespan == 2


@pytest.mark.integration
def test_btpg_solver_marker_ends_in_btpg_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_path = shutil.which("true") or shutil.which("sleep")
    if sleep_path is None:
        pytest.skip("no executable available to satisfy _check_binary_exists")

    plan = Plan.from_paths(
        agents=[Agent(id=0, start=(0, 0), goal=(0, 0))],
        paths=[PlanPath(agent_id=0, vertices=[(0, 0)])],
    )
    fixture_payload = (FIXTURES / "btpg_sample.json").read_text(encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_runner(fixture_payload))

    wrapper = BTPGMaxBaseline(binary_path=sleep_path)
    result = wrapper.post_process(plan)
    assert result.solver_used.endswith("+btpg_max")
