"""Unit tests for slackcertify.core.tpg."""

from __future__ import annotations

import pytest

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Path, Plan
from slackcertify.core.tpg import TPGNode, build_tpg


@pytest.fixture
def grid() -> GridGraph:
    return GridGraph(4, 4, set())


def _two_agent_shared_vertex_plan() -> Plan:
    """Same hand-built plan as in test_conflict.py: cross at (1, 1) at t=2."""
    agents = [
        Agent(id=0, start=(0, 0), goal=(2, 1)),
        Agent(id=1, start=(1, 3), goal=(1, 0)),
    ]
    paths = [
        Path(agent_id=0, vertices=[(0, 0), (1, 0), (1, 1), (2, 1)]),
        Path(agent_id=1, vertices=[(1, 3), (1, 2), (1, 1), (1, 0)]),
    ]
    return Plan.from_paths(agents, paths)


@pytest.mark.unit
def test_nominal_tpg_has_all_nodes() -> None:
    plan = _two_agent_shared_vertex_plan()
    tpg = build_tpg(plan)
    expected_nodes = {
        TPGNode(0, (0, 0), 0),
        TPGNode(0, (1, 0), 1),
        TPGNode(0, (1, 1), 2),
        TPGNode(0, (2, 1), 3),
        TPGNode(1, (1, 3), 0),
        TPGNode(1, (1, 2), 1),
        TPGNode(1, (1, 1), 2),
        TPGNode(1, (1, 0), 3),
    }
    assert set(tpg.nodes()) == expected_nodes


@pytest.mark.unit
def test_nominal_tpg_is_acyclic() -> None:
    plan = _two_agent_shared_vertex_plan()
    tpg = build_tpg(plan)
    assert tpg.is_acyclic()


@pytest.mark.unit
def test_topological_order_respects_edges() -> None:
    plan = _two_agent_shared_vertex_plan()
    tpg = build_tpg(plan)
    order = tpg.topological_order()
    pos = {n: i for i, n in enumerate(order)}
    for u, v in tpg.graph.edges:
        assert pos[u] < pos[v], f"edge {u} -> {v} violates topological order"


@pytest.mark.unit
def test_linear_extension_strictly_increasing_along_edges() -> None:
    plan = _two_agent_shared_vertex_plan()
    tpg = build_tpg(plan)
    ext = tpg.linear_extension()
    # Each value's integer part equals its node's time_step.
    for node, val in ext.items():
        assert int(val) == node.time_step
    # Every edge has strictly increasing extension values.
    for u, v in tpg.graph.edges:
        assert ext[u] < ext[v], f"linear extension fails monotonicity on {u} -> {v}"


@pytest.mark.unit
def test_repair_edge_round_trip() -> None:
    plan = _two_agent_shared_vertex_plan()
    tpg = build_tpg(plan)
    src = TPGNode(0, (1, 1), 2)
    dst = TPGNode(1, (1, 1), 2)
    # The Type-2 ordering already added src -> dst; adding a repair edge in the
    # same direction is harmless and is recorded under .repair_edges.
    tpg.add_repair_edge(src, dst)
    assert (src, dst) in tpg.repair_edges
    assert tpg.is_acyclic()

    with pytest.raises(KeyError):
        tpg.add_repair_edge(TPGNode(0, (99, 99), 0), dst)


@pytest.mark.unit
def test_topological_order_raises_on_cycle() -> None:
    plan = _two_agent_shared_vertex_plan()
    tpg = build_tpg(plan)
    src = TPGNode(0, (1, 1), 2)
    dst = TPGNode(1, (1, 1), 2)
    # Adding the *reverse* of the existing Type-2 edge introduces a cycle.
    tpg.add_repair_edge(dst, src)
    assert not tpg.is_acyclic()
    with pytest.raises(ValueError, match="cycles"):
        tpg.topological_order()


@pytest.mark.unit
def test_to_dot_smoke(grid: GridGraph) -> None:
    plan = _two_agent_shared_vertex_plan()
    dot = build_tpg(plan).to_dot()
    assert dot.startswith("digraph TPG {")
    assert "type1" in dot
    assert "type2" in dot
