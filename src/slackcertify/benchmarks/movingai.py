"""MovingAI .map / .scen parsers and instance constructor.

The MovingAI benchmark suite ships a textual ``.map`` format (which we
already parse via :meth:`slackcertify.core.graph.GridGraph.from_movingai`)
and a tab-separated ``.scen`` companion file that lists start / goal
pairs grouped into difficulty buckets. This module exposes the
parsers and a single-shot :func:`make_instance` constructor that
returns a ``(graph, agents)`` pair ready to feed into the certifier.

Examples
--------
>>> from pathlib import Path
>>> base = Path(__file__).resolve().parents[3] / "tests" / "data"
>>> graph, agents = make_instance(
...     base / "tiny_4x4.map", base / "tiny_4x4-random-1.scen", n_agents=2
... )
>>> graph.width, len(agents)
(4, 2)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent

__all__ = ["ScenRow", "make_instance", "parse_map_file", "parse_scen_file"]


@dataclass(frozen=True, slots=True)
class ScenRow:
    """One row of a MovingAI ``.scen`` file."""

    bucket: int
    map_name: str
    map_width: int
    map_height: int
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int
    optimal_length: float


def parse_map_file(path: str | Path) -> GridGraph:
    """Parse a MovingAI ``.map`` file into a :class:`GridGraph`.

    Thin alias for :meth:`GridGraph.from_movingai`; the indirection
    gives the :mod:`slackcertify.benchmarks` namespace a self-contained
    public surface.

    Examples
    --------
    >>> from pathlib import Path
    >>> p = Path(__file__).resolve().parents[3] / "tests" / "data" / "tiny_4x4.map"
    >>> parse_map_file(p).width
    4
    """
    return GridGraph.from_movingai(path)


def parse_scen_file(path: str | Path) -> list[ScenRow]:
    """Parse a MovingAI ``.scen`` file into a list of :class:`ScenRow`.

    Skips the leading ``version`` header, blank lines, and lines
    starting with ``#``. Each data line must have nine tab-separated
    columns; malformed rows raise :class:`SyntaxError` with the
    offending 1-based line number.

    Examples
    --------
    >>> from pathlib import Path
    >>> p = (Path(__file__).resolve().parents[3] / "tests" / "data"
    ...      / "tiny_4x4-random-1.scen")
    >>> rows = parse_scen_file(p)
    >>> len(rows), rows[0].start_x, rows[0].goal_y
    (3, 0, 3)
    """
    rows: list[ScenRow] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("version"):
            continue
        cols = line.split("\t")
        if len(cols) != 9:
            raise SyntaxError(f"{path}:{lineno}: expected 9 tab-separated columns, got {len(cols)}")
        try:
            rows.append(
                ScenRow(
                    bucket=int(cols[0]),
                    map_name=cols[1],
                    map_width=int(cols[2]),
                    map_height=int(cols[3]),
                    start_x=int(cols[4]),
                    start_y=int(cols[5]),
                    goal_x=int(cols[6]),
                    goal_y=int(cols[7]),
                    optimal_length=float(cols[8]),
                )
            )
        except ValueError as exc:
            raise SyntaxError(f"{path}:{lineno}: {exc}") from exc
    return rows


def make_instance(
    map_path: str | Path,
    scen_path: str | Path,
    n_agents: int,
    scen_index: int = 0,
) -> tuple[GridGraph, list[Agent]]:
    """Build a ``(graph, agents)`` pair from a MovingAI map + scen.

    Selects rows ``scen[scen_index : scen_index + n_agents]``,
    validates their start / goal coordinates against the parsed graph,
    and returns :class:`Agent` instances with ids ``0..n-1``.

    Raises
    ------
    ValueError
        If the scen file has fewer than ``scen_index + n_agents`` rows,
        or if any agent's start / goal lies on an obstacle or
        out-of-bounds cell.

    Examples
    --------
    >>> from pathlib import Path
    >>> base = Path(__file__).resolve().parents[3] / "tests" / "data"
    >>> _, agents = make_instance(base / "tiny_4x4.map",
    ...                           base / "tiny_4x4-random-1.scen", n_agents=2)
    >>> [a.id for a in agents]
    [0, 1]
    """
    if n_agents < 0:
        raise ValueError(f"n_agents must be non-negative, got {n_agents}")
    if scen_index < 0:
        raise ValueError(f"scen_index must be non-negative, got {scen_index}")

    graph = parse_map_file(map_path)
    rows = parse_scen_file(scen_path)
    end = scen_index + n_agents
    if end > len(rows):
        raise ValueError(f"scen has only {len(rows)} rows; need scen_index+n_agents = {end}")

    agents: list[Agent] = []
    selected: Sequence[ScenRow] = rows[scen_index:end]
    for i, row in enumerate(selected):
        start = (row.start_x, row.start_y)
        goal = (row.goal_x, row.goal_y)
        if not graph.is_free(start):
            raise ValueError(f"agent {i} start {start} on obstacle (or out of bounds)")
        if not graph.is_free(goal):
            raise ValueError(f"agent {i} goal {goal} on obstacle (or out of bounds)")
        agents.append(Agent(id=i, start=start, goal=goal))
    return graph, agents
