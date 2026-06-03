"""Unit tests for slackcertify.ilp.encoding.

Skips at module-load time when the ``[ilp]`` optional extras are
absent (no PuLP). ``build_wait_insertion_ilp`` is imported lazily
inside each test body so the module's top of file remains
import-statement-only — there's no E402 to suppress.
"""

from __future__ import annotations

import pytest

from slackcertify.core.plan import Agent, Path, Plan

pulp = pytest.importorskip("pulp")


def _two_agent_disjoint_plan() -> Plan:
    """Two agents on 5-cell straight lines that never share a vertex."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(4, 0)),
        Agent(id=1, start=(0, 1), goal=(4, 1)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),
        Path(agent_id=1, vertices=[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
    ]
    return Plan.from_paths(agents, paths)


def _two_agent_one_vertex_conflict_plan() -> Plan:
    """Crafted so the only delta=1 unsafe pair is one vertex conflict at (1, 0).

    Agent 0 walks east; agent 1 enters (1, 0) from the north two steps later
    and continues to its goal at (0, 0). The 2-step gap on the only shared
    edge keeps the edge-swap detector silent at delta=1, and neither agent's
    goal padding lands on a cell the other visits.
    """
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 0)),
        Agent(id=1, start=(1, 2), goal=(0, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)]),
        Path(agent_id=1, vertices=[(1, 2), (1, 1), (1, 0), (0, 0)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_build_ilp_creates_wait_vars_per_agent_visit() -> None:
    from slackcertify.ilp.encoding import build_wait_insertion_ilp

    plan = _two_agent_disjoint_plan()
    enc = build_wait_insertion_ilp(plan, delta=0)
    # 5 visits per agent x 2 agents = 10 wait variables.
    assert len(enc.wait_vars) == 10
    assert {key[0] for key in enc.wait_vars} == {0, 1}
    # Objective is the sum of all wait variables.
    obj_terms = enc.problem.objective.to_dict()
    obj_var_names = {term["name"] for term in obj_terms}
    expected = {v.name for v in enc.wait_vars.values()}
    assert obj_var_names == expected


@pytest.mark.unit
def test_build_ilp_constraint_count_matches_same_cell_pairs() -> None:
    """Post-PART-B contract: the encoding produces ``2 * |same-cell pairs|``
    disjunctive constraints, not ``2 * |nominal conflicts|``. The previous
    semantics was unsound because wait shifts could induce post-shift
    conflicts the LP wasn't constraining (see the corridor-swap diagnostic)."""
    from slackcertify.ilp.encoding import build_wait_insertion_ilp

    plan = _two_agent_one_vertex_conflict_plan()
    enc = build_wait_insertion_ilp(plan, delta=1)

    # Sanity: still one nominal conflict (the t_i=1 / t_j=2 visit pair at (1, 0)).
    assert len(enc.conflicts) == 1

    # Count same-cell pairs across distinct agents directly so this test
    # documents the new invariant rather than re-implementing the encoding.
    cells_by_agent = {p.agent_id: set(p.vertices) for p in plan.paths}
    shared_cells = cells_by_agent[0] & cells_by_agent[1]
    expected_pairs = len(shared_cells)  # exactly one cross-agent pair per shared cell here
    assert len(enc.problem.constraints) == 2 * expected_pairs
    assert expected_pairs >= len(enc.conflicts)  # broader set covers the nominal one


@pytest.mark.unit
def test_build_ilp_handles_zero_conflicts_cleanly() -> None:
    from slackcertify.ilp.encoding import build_wait_insertion_ilp

    plan = _two_agent_disjoint_plan()
    enc = build_wait_insertion_ilp(plan, delta=0)
    assert enc.conflicts == []
    # No conflict constraints; only the objective row.
    assert len(enc.problem.constraints) == 0
    # Every wait variable's upper bound is delta * len(conflicts) + 1 = 1.
    for v in enc.wait_vars.values():
        assert v.upBound == 1


@pytest.mark.unit
def test_build_ilp_raises_on_invalid_delta() -> None:
    from slackcertify.ilp.encoding import build_wait_insertion_ilp

    plan = _two_agent_disjoint_plan()
    with pytest.raises(ValueError, match="non-negative"):
        build_wait_insertion_ilp(plan, delta=-1)


@pytest.mark.unit
def test_ilp_constraints_cover_all_same_cell_pairs() -> None:
    """Two agents traverse cells (0, 0)..(0, 3) in opposite directions but
    with a 5-step time gap, so the nominal plan is conflict-free at delta=1.
    The encoding must still constrain every shared cell — otherwise wait
    shifts could induce new conflicts the LP wouldn't notice (the bug
    surfaced by the corridor-swap diagnostic)."""
    from slackcertify.core.conflict import detect_conflicts
    from slackcertify.ilp.encoding import build_wait_insertion_ilp

    agents = [
        Agent(id=0, start=(0, 0), goal=(1, 3)),
        Agent(id=1, start=(0, 8), goal=(0, 0)),
    ]
    paths = [
        # Agent 0 walks north along column 0 then exits east at the top.
        Path(agent_id=0, vertices=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)]),
        # Agent 1 walks south down column 0 reaching the four shared cells
        # at t = 5, 6, 7, 8 — five steps after agent 0 left them.
        Path(
            agent_id=1,
            vertices=[
                (0, 8),
                (0, 7),
                (0, 6),
                (0, 5),
                (0, 4),
                (0, 3),
                (0, 2),
                (0, 1),
                (0, 0),
            ],
        ),
    ]
    plan = Plan.from_paths(agents, paths)

    # Sanity: the nominal plan really is conflict-free at delta=1 — so the
    # only thing forcing constraints is the broader same-cell-pair coverage.
    assert detect_conflicts(plan, delta=1) == []

    enc = build_wait_insertion_ilp(plan, delta=1)
    # 4 shared cells, 1 cross-agent pair per cell, 2 disjunctive constraints
    # per pair = 8 constraints minimum.
    assert len(enc.problem.constraints) >= 8
