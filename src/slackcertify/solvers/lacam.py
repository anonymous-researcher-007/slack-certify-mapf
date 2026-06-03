"""LaCAM* anytime subprocess wrapper.

The upstream binary (Kei18/lacam2) writes its solution to the ``-o`` output
file, not stdout; stdout carries only a one-line summary. The output file is
a key=value block followed by a ``solution=`` section, one line per timestep,
each agent's position in agent-id order, comma-separated::

    agents=3
    ...
    solution=
    0:(0,0),(0,2),(0,1),
    1:(0,1),(0,3),(0,2),
    ...

The wrapper overrides :meth:`solve` to read the ``-o`` file and pass its
contents to :meth:`_parse_output`, so :meth:`_parse_output` remains a pure
text parser (callable directly in tests). The binary is invoked with ``-i``
(scenario), ``-O 2`` (sum-of-loss, the eventually optimal LaCAM* mode), and
``-o`` (output file).
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

__all__ = ["LaCAMStarSolver"]

_TIMESTEP_LINE_RE = re.compile(r"^(\d+)\s*:\s*(.+)$")
_VERTEX_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


class LaCAMStarSolver(BinarySolverBase):
    """Wrapper around the upstream LaCAM* binary (Kei18/lacam2)."""

    NAME: ClassVar[str] = "lacam_star"

    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        """Construct the LaCAM* argv vector for the subprocess call."""
        output_file = tmpdir / "lacam_output.txt"
        return [
            str(self.binary_path),
            "-m",
            str(map_path),
            "-i",  # scenario file (upstream flag is -i/--scen)
            str(scen_path),
            "-N",
            str(n_agents),
            "-t",
            f"{time_limit_s:g}",
            "-O",  # objective 2 = sum_of_loss = eventually-optimal LaCAM*
            "2",
            "-o",
            str(output_file),
        ]

    def solve(
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        time_limit_s: float,
        **kwargs: object,
    ) -> Plan:
        """Solve via the LaCAM* binary, reading the solution from the -o file.

        Overrides :meth:`BinarySolverBase.solve` because LaCAM* writes its
        solution to the ``-o`` output file rather than stdout. We read that
        file's contents inside the tmpdir context and hand them to
        :meth:`_parse_output` (which therefore parses solver *output text*,
        regardless of whether a given solver emits it on stdout or to a file).
        """
        self._check_binary_exists()
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            map_path, scen_path = self._write_movingai_input(graph, agents, tmpdir)
            output_file = tmpdir / "lacam_output.txt"
            args = self._build_args(
                map_path=map_path,
                scen_path=scen_path,
                n_agents=len(agents),
                time_limit_s=time_limit_s,
                tmpdir=tmpdir,
                **kwargs,
            )
            self._run_subprocess(args, time_limit_s=time_limit_s)
            if not output_file.exists():
                raise SolverOutputParseError(
                    f"LaCAM* did not produce its output file at {output_file}"
                )
            solution_text = output_file.read_text(encoding="utf-8")
            return self._parse_output(solution_text, agents)

    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        """Parse LaCAM* solution text (the ``solution=`` block) into a Plan.

        Parses the text it is given (the -o file contents, passed by
        :meth:`solve`). Header lines (``agents=3``, ``starts=...``,
        ``goals=...``, ``solution=``) do not start with ``<digits>:`` and are
        skipped by ``_TIMESTEP_LINE_RE``.
        """
        timestep_positions: dict[int, list[Cell]] = {}
        for line in stdout.splitlines():
            match = _TIMESTEP_LINE_RE.match(line.strip())
            if not match:
                continue
            t = int(match.group(1))
            positions = [
                (int(m.group(1)), int(m.group(2)))
                for m in _VERTEX_RE.finditer(match.group(2))
            ]
            timestep_positions[t] = positions

        if not timestep_positions:
            raise SolverOutputParseError(
                "LaCAM* output contained no '<t>: <positions>' lines"
            )

        n = len(agents)
        for t, positions in timestep_positions.items():
            if len(positions) != n:
                raise SolverOutputParseError(
                    f"LaCAM* timestep {t} has {len(positions)} positions, expected {n}"
                )

        sorted_ts = sorted(timestep_positions)
        per_agent: list[list[Cell]] = [[] for _ in range(n)]
        for t in sorted_ts:
            for i, cell in enumerate(timestep_positions[t]):
                per_agent[i].append(cell)

        ordered_agents = list(agents)
        plan_paths = [
            PlanPath(agent_id=ordered_agents[i].id, vertices=per_agent[i])
            for i in range(n)
        ]
        return Plan.from_paths(ordered_agents, plan_paths)
