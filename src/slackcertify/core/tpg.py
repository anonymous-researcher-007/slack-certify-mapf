"""Temporal Plan Graph (TPG) following Hönig et al., IEEE RA-L 4(2):1125–1131 (2019).

A TPG node is the triple ``(agent_id, vertex, time_step)``. The graph carries
two kinds of structural edges:

* **Type-1** (intra-agent): consecutive nodes within an agent's path,
  enforcing the per-agent execution order.
* **Type-2** (inter-agent): for every pair of agents that visit the same
  vertex, an edge from the *earlier* visitor to the *later* visitor;
  ties on ``time_step`` are broken by ``agent_id`` ascending.

Repair edges added by the slack certifier are tracked separately so that
they can be inspected, removed, or written into the certificate.

Examples
--------
>>> from slackcertify.core.plan import Agent, Path, Plan
>>> agents = [Agent(id=0, start=(0, 0), goal=(1, 0))]
>>> paths = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
>>> tpg = build_tpg(Plan.from_paths(agents, paths))
>>> tpg.is_acyclic()
True
>>> [n.time_step for n in tpg.topological_order()]
[0, 1]
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import networkx as nx

from slackcertify.core.graph import Cell
from slackcertify.core.plan import Plan

__all__ = ["TPGNode", "TPG", "build_tpg"]


class TPGNode(NamedTuple):
    """Hashable triple identifying a single visit in a temporal plan graph."""

    agent_id: int
    vertex: Cell
    time_step: int


class TPG:
    """Temporal Plan Graph wrapping a :class:`networkx.DiGraph`.

    Parameters
    ----------
    plan
        The MAPF plan from which the nominal TPG is constructed.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> TPG(Plan.from_paths(a, p)).is_acyclic()
    True
    """

    EDGE_TYPE_1: str = "type1"
    EDGE_TYPE_2: str = "type2"
    EDGE_REPAIR: str = "repair"

    def __init__(self, plan: Plan) -> None:
        """Build the nominal TPG (Type-1 + Type-2 edges) from ``plan``."""
        self.plan: Plan = plan
        self.graph: nx.DiGraph = nx.DiGraph()
        self._repair_edges: list[tuple[TPGNode, TPGNode]] = []
        self._build_nominal()

    # ----------------------------------------------------------- construction

    def _build_nominal(self) -> None:
        """Populate the graph with per-step (Type-1) and ordering (Type-2) edges."""
        # Add every (agent, vertex, t) node.
        for path in self.plan.paths:
            for t, v in enumerate(path.vertices):
                self.graph.add_node(TPGNode(path.agent_id, v, t))

        # Type-1: per-agent intra-path edges.
        for path in self.plan.paths:
            verts = path.vertices
            for t in range(len(verts) - 1):
                a = TPGNode(path.agent_id, verts[t], t)
                b = TPGNode(path.agent_id, verts[t + 1], t + 1)
                self.graph.add_edge(a, b, kind=self.EDGE_TYPE_1)

        # Type-2: at every shared vertex, earlier-visit -> later-visit.
        # A "visit" is a (vertex, time_step, agent_id) triple. We sort all
        # visits to a given vertex by (time_step, agent_id) ascending and
        # connect each consecutive pair belonging to *different* agents.
        per_vertex: dict[Cell, list[TPGNode]] = {}
        for path in self.plan.paths:
            for t, v in enumerate(path.vertices):
                per_vertex.setdefault(v, []).append(TPGNode(path.agent_id, v, t))

        for visits in per_vertex.values():
            visits.sort(key=lambda n: (n.time_step, n.agent_id))
            # Connect every pair (i, j) with i earlier than j and i.agent != j.agent.
            for i_idx, ni in enumerate(visits):
                for nj in visits[i_idx + 1 :]:
                    if ni.agent_id == nj.agent_id:
                        continue
                    self.graph.add_edge(ni, nj, kind=self.EDGE_TYPE_2)

    # ------------------------------------------------------------- mutations

    def add_repair_edge(self, from_node: TPGNode, to_node: TPGNode) -> None:
        """Add a directed *repair* edge to the TPG.

        The edge is recorded both in the underlying graph (with
        ``kind="repair"``) and in :attr:`repair_edges` so the certifier
        can later enumerate the precise set of edges it introduced.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0)),
        ...      Agent(id=1, start=(1, 0), goal=(0, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)]),
        ...      Path(agent_id=1, vertices=[(1, 0), (0, 0)])]
        >>> tpg = build_tpg(Plan.from_paths(a, p))
        >>> tpg.add_repair_edge(TPGNode(0, (1, 0), 1), TPGNode(1, (0, 0), 1))
        >>> tpg.repair_edges == [
        ...     (TPGNode(0, (1, 0), 1), TPGNode(1, (0, 0), 1)),
        ... ]
        True
        """
        if from_node not in self.graph:
            raise KeyError(f"from_node {from_node} not in TPG")
        if to_node not in self.graph:
            raise KeyError(f"to_node {to_node} not in TPG")
        self.graph.add_edge(from_node, to_node, kind=self.EDGE_REPAIR)
        self._repair_edges.append((from_node, to_node))

    @property
    def repair_edges(self) -> list[tuple[TPGNode, TPGNode]]:
        """Return the repair edges added since construction (in insertion order)."""
        return list(self._repair_edges)

    # ----------------------------------------------------------- queries

    def is_acyclic(self) -> bool:
        """Return ``True`` if the TPG is a DAG (i.e. globally consistent)."""
        return bool(nx.is_directed_acyclic_graph(self.graph))

    def topological_order(self) -> list[TPGNode]:
        """Return one topological order of the TPG nodes.

        Raises
        ------
        ValueError
            If the TPG contains a cycle.
        """
        if not self.is_acyclic():
            raise ValueError("topological_order requested on a TPG with cycles")
        return list(nx.topological_sort(self.graph))

    def linear_extension(self) -> dict[TPGNode, float]:
        """Return the ``t + i / (n + 1)`` linear extension used in Lemma 1.

        The mapping is strictly increasing along every edge of the TPG and
        its integer part equals the node's ``time_step``. With ``n``
        nodes, the fractional offset ``i / (n + 1) ∈ (0, 1)`` is
        guaranteed not to bridge an integer boundary, so ordering at the
        same time-step is preserved.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(1, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0)])]
        >>> tpg = build_tpg(Plan.from_paths(a, p))
        >>> ext = tpg.linear_extension()
        >>> sorted(ext.values())
        [0.3333333333333333, 1.6666666666666667]
        """
        order = self.topological_order()
        n = len(order)
        denom = n + 1
        out: dict[TPGNode, float] = {}
        for i, node in enumerate(order, start=1):
            out[node] = node.time_step + i / denom
        return out

    def predecessors_at_vertex(self, agent_id: int, vertex: Cell) -> list[TPGNode]:
        """Return TPG predecessors of every visit of ``agent_id`` to ``vertex``.

        Aggregates :meth:`networkx.DiGraph.predecessors` over all nodes
        ``(agent_id, vertex, t)`` for any ``t``. Predecessors are
        deduplicated and returned in deterministic order (sorted by
        ``(agent_id, time_step)``).

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
        >>> tpg = build_tpg(Plan.from_paths(a, p))
        >>> tpg.predecessors_at_vertex(0, (2, 0))
        [TPGNode(agent_id=0, vertex=(1, 0), time_step=1)]
        """
        preds: set[TPGNode] = set()
        for n in self.graph.nodes:
            if n.agent_id == agent_id and n.vertex == vertex:
                preds.update(self.graph.predecessors(n))
        return sorted(preds, key=lambda nn: (nn.agent_id, nn.time_step))

    # ------------------------------------------------------------- viz

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation for debug visualisation.

        Examples
        --------
        >>> from slackcertify.core.plan import Agent, Path, Plan
        >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
        >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
        >>> "digraph TPG" in build_tpg(Plan.from_paths(a, p)).to_dot()
        True
        """
        lines: list[str] = ["digraph TPG {", '    rankdir="LR";']
        node_id: dict[TPGNode, str] = {}
        for i, n in enumerate(self.graph.nodes):
            nid = f"n{i}"
            node_id[n] = nid
            label = f"a{n.agent_id} {n.vertex} t={n.time_step}"
            lines.append(f'    {nid} [label="{label}"];')
        for u, v, data in self.graph.edges(data=True):
            kind = str(data.get("kind", ""))
            style = {
                self.EDGE_TYPE_1: "solid",
                self.EDGE_TYPE_2: "dashed",
                self.EDGE_REPAIR: "dotted",
            }.get(kind, "solid")
            lines.append(f'    {node_id[u]} -> {node_id[v]} [style={style}, label="{kind}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------- containment

    def __contains__(self, node: object) -> bool:
        """Return ``True`` iff ``node`` is a vertex of this TPG."""
        return bool(self.graph.has_node(node))

    def nodes(self) -> Iterable[TPGNode]:
        """Iterate over all TPG nodes."""
        return iter(list(self.graph.nodes))


def build_tpg(plan: Plan) -> TPG:
    """Construct the nominal :class:`TPG` for ``plan``.

    Examples
    --------
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> a = [Agent(id=0, start=(0, 0), goal=(0, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0)])]
    >>> isinstance(build_tpg(Plan.from_paths(a, p)), TPG)
    True
    """
    return TPG(plan)
