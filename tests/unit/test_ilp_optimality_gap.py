"""Slow optimality-gap test: ILP vs. SlackCertify on a 4-agent instance.

Skips at module-load time when the ``[ilp]`` optional extras are
absent (no PuLP). ``solve_optimal_wait_insertion`` is imported
lazily inside the test body so the module's top of file remains
import-statement-only — there's no E402 to suppress.
"""

from __future__ import annotations

from pathlib import Path as _Path

import pytest

from slackcertify.benchmarks.scenarios import ScenarioRegistry
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.ilp._solver_detection import available_solvers
from slackcertify.repair import slack_certify
from slackcertify.solvers._fake import FakeSolver

pulp = pytest.importorskip("pulp")

_AVAILABLE = available_solvers()
_REASON = "no ILP solver installed; install slackcertify[ilp]"


def _fallback_4_agent_plan() -> Plan:
    """7×7 fallback when the MovingAI suite isn't fetched.

    Agent 0 walks east through three "hub" cells along row 0 at nominal
    times t=1, 3, 5; agents 1, 2, 3 each meet agent 0 at exactly one
    hub at t=2, 4, 6 respectively. No head-on swaps; resolvable by
    wait insertion alone, so both SlackCertify and the ILP converge.
    """
    agents = [
        Agent(id=0, start=(0, 0), goal=(6, 0)),
        Agent(id=1, start=(1, 2), goal=(1, 0)),
        Agent(id=2, start=(3, 4), goal=(3, 0)),
        Agent(id=3, start=(5, 6), goal=(5, 0)),
    ]
    paths = [
        Path(
            agent_id=0,
            vertices=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],
        ),
        Path(agent_id=1, vertices=[(1, 2), (1, 1), (1, 0)]),
        Path(agent_id=2, vertices=[(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)]),
        Path(
            agent_id=3,
            vertices=[(5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0)],
        ),
    ]
    return Plan.from_paths(agents, paths)


def _try_movingai_4_agent_plan() -> Plan | None:
    """Try the random-32-32-20 benchmark; return None if it isn't fetched."""
    try:
        registry = ScenarioRegistry()
        graph, agents = registry.load_instance(map_name="random-32-32-20", n_agents=4, scen_seed=1)
    except FileNotFoundError:
        return None
    plan = FakeSolver().solve(graph, agents, time_limit_s=0.0)
    return plan


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_optimality_gap_on_random_instance(capsys: pytest.CaptureFixture[str]) -> None:
    from slackcertify.ilp.solver import solve_optimal_wait_insertion

    plan = _try_movingai_4_agent_plan() or _fallback_4_agent_plan()

    sc_plan, sc_cert = slack_certify(plan, mode="bounded", delta=2, max_outer_rounds=64)
    sc_waits = sc_cert.total_wait_inserted

    ilp = solve_optimal_wait_insertion(plan, delta=2, time_limit_s=60.0)
    assert ilp.objective_value is not None, f"ILP did not return an objective: {ilp}"

    ilp_waits = ilp.objective_value
    # ILP is exact, so it must match or beat the greedy.
    assert ilp_waits <= sc_waits, (
        f"ILP optimum ({ilp_waits}) is worse than SlackCertify's greedy "
        f"({sc_waits}); this contradicts ILP optimality."
    )

    if ilp_waits == 0:
        gap_pct = 0.0
    else:
        gap_pct = 100.0 * (sc_waits - ilp_waits) / max(ilp_waits, 1)
    print(
        f"Optimality gap: SlackCertify={sc_waits}, ILP={ilp_waits}, "
        f"gap=(SC-ILP)/ILP={gap_pct:.2f}%"
    )
    # Force the print into the captured stream so reviewers see it.
    capsys.readouterr()


# Re-export Path for tooling that walks the module.
_ = _Path
