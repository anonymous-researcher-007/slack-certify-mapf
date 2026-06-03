"""EECBS subprocess wrapper.

The wrapper invokes the EECBS binary with the standard CLI flags
``-m / -a / -k / --suboptimality / -o / --outputPaths / -t``. EECBS writes
its CSV statistics to ``-o`` but the per-agent paths to a *separate*
``--outputPaths`` file; :meth:`EECBSSolver.solve` therefore overrides the base
to read that paths file (not stdout) and parse the per-agent plain-text path
output of the form::

    Agent 0: (0,0)->(1,0)->(2,0)
    Agent 1: (3,0)->(2,0)->(1,0)

into a :class:`~slackcertify.core.plan.Plan`.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from slackcertify.core.graph import Cell, GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.core.plan import Path as PlanPath
from slackcertify.solvers.base import BinarySolverBase, SolverOutputParseError

__all__ = ["EECBSSolver"]


_AGENT_LINE_RE = re.compile(r"Agent\s+(\d+)\s*:\s*(.+)$")
_VERTEX_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


class EECBSSolver(BinarySolverBase):
    """Wrapper around the upstream EECBS binary."""

    NAME: ClassVar[str] = "eecbs"

    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        """Construct the EECBS argv vector for the subprocess call."""
        raw = kwargs.get("suboptimality", 1.2)
        if not isinstance(raw, (int, float)):
            raise TypeError(f"suboptimality must be a number, got {type(raw).__name__}")
        suboptimality = float(raw)
        # EECBS writes CSV statistics to -o, but the per-agent paths
        # ("Agent N: (x,y)->...") go to a SEPARATE --outputPaths file. We need
        # both; solve() reads the paths file (matching this fixed name).
        stats_file = tmpdir / "eecbs_output.txt"
        paths_file = tmpdir / "eecbs_paths.txt"
        return [
            str(self.binary_path),
            "-m",
            str(map_path),
            "-a",
            str(scen_path),
            "-k",
            str(n_agents),
            "--suboptimality",
            f"{suboptimality:g}",
            "-o",
            str(stats_file),
            "--outputPaths",
            str(paths_file),
            "-t",
            f"{time_limit_s:g}",
        ]

    def solve(
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        time_limit_s: float,
        **kwargs: object,
    ) -> Plan:
        """Solve via the EECBS binary, reading paths from the ``--outputPaths`` file.

        Overrides :meth:`BinarySolverBase.solve` because EECBS writes its
        per-agent paths to the ``--outputPaths`` file, not stdout (stdout and
        ``-o`` carry only the CSV statistics summary). We read that file's
        contents inside the tmpdir context and hand them to
        :meth:`_parse_output`, exactly as :class:`LaCAMStarSolver` does for its
        ``-o`` file.
        """
        self._check_binary_exists()
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            map_path, scen_path = self._write_movingai_input(graph, agents, tmpdir)
            paths_file = tmpdir / "eecbs_paths.txt"
            args = self._build_args(
                map_path=map_path,
                scen_path=scen_path,
                n_agents=len(agents),
                time_limit_s=time_limit_s,
                tmpdir=tmpdir,
                **kwargs,
            )
            self._run_subprocess(args, time_limit_s=time_limit_s)
            if not paths_file.exists():
                raise SolverOutputParseError(
                    f"EECBS did not produce its paths file at {paths_file}"
                )
            paths_text = paths_file.read_text(encoding="utf-8")
            return self._parse_output(paths_text, agents)

    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        """Parse EECBS's ``Agent N: (row,col)->...`` path text into a :class:`Plan`.

        Parses the text it is given — the ``--outputPaths`` file contents passed
        by :meth:`solve` (not subprocess stdout, which only carries the CSV
        statistics).

        Coordinate convention. EECBS indexes cells internally as
        ``(row, col)`` and prints them in that order, e.g. it emits ``(1,24)``
        for the cell whose MovingAI ``.scen`` start is ``col=24, row=1``. The
        rest of the codebase — the ``.scen`` the wrapper writes, the
        :class:`~slackcertify.core.plan.Agent` start/goal, and the LaCAM*
        parser — uses ``(col, row) == (x, y)``. We therefore **swap** EECBS's
        two integers on parse (``(row, col) -> (col, row)``) so plans from EECBS
        and LaCAM* share one convention and the certifier sees geometrically
        correct cells. (The ``_VERTEX_RE`` regex itself is unchanged; only the
        order in which the two captured integers are assigned to ``(x, y)``.)
        """
        paths_by_id: dict[int, list[Cell]] = {}
        for line in stdout.splitlines():
            match = _AGENT_LINE_RE.match(line.strip())
            if not match:
                continue
            agent_id = int(match.group(1))
            # EECBS prints (row, col); swap to the codebase's (col, row)=(x, y).
            verts = [
                (int(m.group(2)), int(m.group(1))) for m in _VERTEX_RE.finditer(match.group(2))
            ]
            if not verts:
                raise SolverOutputParseError(
                    f"EECBS output line for agent {agent_id} has no vertices: {line!r}"
                )
            paths_by_id[agent_id] = verts

        if not paths_by_id:
            raise SolverOutputParseError("EECBS output contained no 'Agent N:' lines")

        ordered_agents = list(agents)
        plan_paths: list[PlanPath] = []
        for a in ordered_agents:
            if a.id not in paths_by_id:
                raise SolverOutputParseError(f"EECBS output is missing a path for agent {a.id}")
            plan_paths.append(PlanPath(agent_id=a.id, vertices=paths_by_id[a.id]))
        return Plan.from_paths(ordered_agents, plan_paths)
