"""Integration tests for the Kottinger CBS-delay wrapper.

Two layers of coverage:

* Fail-loud tests: pin the wrapper's behaviour when the binary is
  absent (the post-D3 contract — :class:`SolverNotFoundError` is
  raised, no silent fallback). These run in every environment.
* Real-binary test: marked ``skipif(not binary_exists)``. Runs only
  on machines where ``scripts/install_baselines.sh`` has produced
  ``src/slackcertify/solvers/external_bin/kottinger`` from the
  ``aria-systems-group/Delay-Robust-MAPF`` upstream.
"""

from __future__ import annotations

import pytest
from baselines._paths import external_bin
from baselines.kottinger.wrapper import KottingerDelayBaseline

from slackcertify.certify.bounded import is_delta_certified
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath
from slackcertify.solvers.base import SolverNotFoundError

_KOTTINGER_BIN = external_bin("kottinger")
_BINARY_AVAILABLE = _KOTTINGER_BIN.exists()


def _three_agent_plan() -> Plan:
    """Three agents that share (1, 1) at staggered nominal times."""
    agents = [
        Agent(id=0, start=(0, 1), goal=(2, 1)),
        Agent(id=1, start=(1, 0), goal=(1, 2)),
        Agent(id=2, start=(2, 0), goal=(2, 2)),
    ]
    paths = [
        PlanPath(agent_id=0, vertices=[(0, 1), (1, 1), (2, 1)]),
        PlanPath(agent_id=1, vertices=[(1, 0), (1, 1), (1, 2)]),
        PlanPath(agent_id=2, vertices=[(2, 0), (2, 1), (2, 1), (2, 2)]),
    ]
    return Plan.from_paths(agents, paths)


# ------------------------------------------------------------ fail-loud path


@pytest.mark.integration
def test_kottinger_offline_raises_solver_not_found_when_binary_absent() -> None:
    """D3 contract: no silent fallback — the wrapper raises loudly."""
    plan = _three_agent_plan()
    wrapper = KottingerDelayBaseline(binary_path="/nonexistent")
    with pytest.raises(SolverNotFoundError) as excinfo:
        wrapper.solve_offline(plan, delta=1, time_limit_s=1.0)
    # The error must name the binary path and the build hint.
    msg = str(excinfo.value)
    assert "install_baselines.sh kottinger" in msg
    assert "aria-systems-group/Delay-Robust-MAPF" in msg


@pytest.mark.integration
def test_kottinger_offline_error_points_at_canonical_binary_path() -> None:
    """The error message must reference the expected install location."""
    plan = _three_agent_plan()
    wrapper = KottingerDelayBaseline(binary_path="/nonexistent")
    with pytest.raises(SolverNotFoundError) as excinfo:
        wrapper.solve_offline(plan, delta=2, time_limit_s=1.0)
    # The post-D3 message points at the canonical default install path
    # (where the real binary lands) rather than the per-instance
    # binary_path override — the latter is just a test wart.
    assert str(_KOTTINGER_BIN) in str(excinfo.value)


@pytest.mark.integration
def test_kottinger_online_solve_raises_with_clear_message() -> None:
    plan = _three_agent_plan()
    wrapper = KottingerDelayBaseline(binary_path="/nonexistent")
    with pytest.raises(NotImplementedError, match="V critical path"):
        wrapper.solve_online(plan, observed_delay={}, time_limit_s=0.1)


# ------------------------------------------------------------- real binary


@pytest.mark.integration
@pytest.mark.skipif(
    not _BINARY_AVAILABLE,
    reason=(
        "Kottinger binary not built; run scripts/install_baselines.sh "
        "to fetch and build third_party/delay-introduction."
    ),
)
def test_kottinger_offline_real_binary_produces_delta_disjoint_plan() -> None:
    """The upstream binary must produce a Δ-disjoint plan on the smoke fixture.

    This test only runs when ``scripts/install_baselines.sh`` has
    produced the binary at the expected path. In sandboxed
    environments it skips cleanly. The expected smoke runtime is
    well under the 5 s budget on a 3-agent instance.
    """
    plan = _three_agent_plan()
    wrapper = KottingerDelayBaseline()  # default binary_path
    assert wrapper.binary_path == _KOTTINGER_BIN
    result = wrapper.solve_offline(plan, delta=1, time_limit_s=5.0)
    assert is_delta_certified(result.plan, delta=1), (
        "Kottinger upstream binary produced a non-Δ-disjoint plan; "
        "check the wrapper's args/parse TODO_VERIFY placeholders "
        "against the upstream README."
    )
    assert result.solver_used.endswith("+kottinger_offline_delta=1_binary"), (
        f"expected '_binary' provenance suffix, got " f"{result.solver_used!r}"
    )
