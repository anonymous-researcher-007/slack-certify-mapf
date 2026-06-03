"""PIBT subprocess wrapper (Kei18/pibt2, executable ``mapf``).

PIBT is a fast incomplete priority-inheritance solver; typical wall-clock
budgets are 1-5 seconds.

Unlike the MovingAI ``.map`` + ``.scen`` interface used by the base class,
PIBT2 consumes its *own* instance/param file (``-i``) and writes the
solution to the ``-o`` output file (stdout carries only a one-line
summary). The output ``solution=`` block, however, is byte-for-byte the
same timestep-major ``<t>:(x,y),(x,y),...`` format LaCAM* emits, so we
reuse :meth:`LaCAMStarSolver._parse_output` verbatim rather than writing a
second parser.

Two PIBT2-specific details drive the :meth:`solve` override:

* **Param file.** ``-i`` points at a key=value header
  (``map_file=``/``agents=``/``random_problem=0``/``max_timestep=``/
  ``max_comp_time=``) followed by one ``x_s,y_s,x_g,y_g`` line per agent,
  in the order the agents are given. Coordinates are ``(x,y)=(col,row)``
  with a top-left origin -- the same convention as the LaCAM* wrapper.
* **Map resolution.** The binary ignores the directory in ``map_file=`` and
  resolves the *basename* against a directory baked into the executable. See
  :data:`PIBT_MAP_DIR` (resolved from ``PIBT_MAP_DIR`` env, default cwd-relative
  ``map``). The wrapper writes the map there under a unique basename and removes
  it afterwards (the directory may be shared by concurrent workers).

The ``solution=`` block is timestep-major (``<t>:(x,y),(x,y),...``, the i-th
``(x,y)`` is agent i at time t), identical to LaCAM*, so :meth:`_parse_output`
reuses :meth:`LaCAMStarSolver._parse_output` (which transposes columns to
per-agent paths and reads ``(x,y)=(col,row)`` verbatim, no swap).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.solvers.base import (
    BinarySolverBase,
    SolverError,
    SolverOutputParseError,
    SolverTimeoutError,
)
from slackcertify.solvers.lacam import LaCAMStarSolver

__all__ = ["PIBTSolver"]


# Where the Kei18 ``mapf`` binary looks for its map file. This binary build
# IGNORES the directory in ``map_file=`` and resolves the *basename* against a
# directory baked into the executable at compile time (the ``_MAPDIR_`` macro):
#
#   * the upstream/source build resolves the cwd-relative ``map/`` (verified by
#     ``strings`` on the locally-built binary -> ``map/``); the wrapper runs
#     with ``cwd=tmpdir`` so this lands in ``tmpdir/map/``.
#   * the target-machine binary has an ABSOLUTE path baked in
#     (``strings`` -> ``/home/<user>/pibt2/pibt2/../map/`` == ``/home/<user>/pibt2/
#     map/``); ``strace`` confirmed it opens ``/home/<user>/pibt2/map/<basename>``
#     regardless of cwd.
#
# Because the directory is baked into the binary, the wrapper is NON-PORTABLE
# across PIBT builds. We resolve it at runtime: set ``PIBT_MAP_DIR`` to the
# binary's baked-in dir (the target machine sets
# ``PIBT_MAP_DIR=/home/<user>/pibt2/map``); the default ``map`` matches the
# cwd-relative source build. A relative value is taken relative to the per-solve
# tmpdir (and the binary is run with ``cwd=tmpdir``); an absolute value is used
# as-is.
PIBT_MAP_DIR: str = os.environ.get("PIBT_MAP_DIR", "map")


class PIBTSolver(BinarySolverBase):
    """Wrapper around the upstream PIBT binary (Kei18/pibt2, ``-s PIBT``)."""

    NAME: ClassVar[str] = "pibt"
    DEFAULT_TIME_LIMIT_S: ClassVar[float] = 5.0

    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        """Unused: PIBT2's input differs from the base MovingAI interface.

        :meth:`solve` is overridden and constructs the argv itself (the
        base class' MovingAI ``.map`` + ``.scen`` contract does not apply
        to PIBT2, which consumes its own ``-i`` param file). This method
        is required by the ABC but never called.
        """
        raise NotImplementedError(
            "PIBTSolver builds its argv inside solve(); _build_args is unused."
        )

    def solve(
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        time_limit_s: float,
        **kwargs: object,
    ) -> Plan:
        """Solve via the PIBT2 binary (``-s PIBT``), reading the ``-o`` file.

        The binary resolves ``map_file=`` by basename against a baked-in
        directory (see :data:`PIBT_MAP_DIR`), so we write the map there under a
        **unique** basename and set ``map_file=<basename>``. The unique name is
        critical: a sweep runs many concurrent workers that all share one
        (possibly hardcoded, absolute) map directory, so a fixed name would let
        workers clobber each other's map. The map file is removed in a
        ``finally`` block so the shared directory does not accumulate files.
        """
        self._check_binary_exists()
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)

            # Resolve the binary's baked-in map directory. Absolute -> shared
            # dir used as-is (target machine); relative -> under this tmpdir
            # (source build, with cwd=tmpdir).
            map_dir = Path(PIBT_MAP_DIR)
            if not map_dir.is_absolute():
                map_dir = tmpdir / map_dir
            os.makedirs(map_dir, exist_ok=True)

            # Reuse the base MovingAI map writer (writes map+scen into tmpdir;
            # the .scen is unused by PIBT2 and harmlessly ignored), then place
            # the .map text under a UNIQUE basename in the resolved map dir.
            # The unique name lets concurrent workers share one map dir without
            # collisions; the binary supplies the directory, so we pass basename
            # only in the param file.
            src_map, _scen = self._write_movingai_input(graph, agents, tmpdir)
            map_name = f"pibt_{uuid.uuid4().hex}.map"
            map_file = map_dir / map_name
            map_file.write_text(src_map.read_text(encoding="utf-8"), encoding="utf-8")

            try:
                param_path = tmpdir / "instance.param"
                param_path.write_text(
                    self._build_param_file(graph, agents, map_name, time_limit_s),
                    encoding="utf-8",
                )

                output_file = tmpdir / "pibt_output.txt"
                time_limit_ms = max(1, int(round(time_limit_s * 1000)))
                args = [
                    str(self.binary_path),
                    "-i",
                    str(param_path),
                    "-o",
                    str(output_file),
                    "-s",
                    "PIBT",
                    "-T",
                    str(time_limit_ms),
                ]

                self._run_pibt(args, tmpdir=tmpdir, time_limit_s=time_limit_s)

                if not output_file.exists():
                    raise SolverOutputParseError(
                        f"PIBT did not produce its output file at {output_file}"
                    )
                solution_text = output_file.read_text(encoding="utf-8")
                self._check_solved(solution_text)
                return self._parse_output(solution_text, agents)
            finally:
                # Always remove our map from the (possibly shared) map dir.
                map_file.unlink(missing_ok=True)

    def _build_param_file(
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        map_name: str,
        time_limit_s: float,
    ) -> str:
        """Render PIBT2's instance/param file for the given instance.

        ``random_problem=0`` forces use of the explicit start/goal lines.
        Coordinates are written as ``(x,y)=(col,row)`` -- ``agent.start``
        and ``agent.goal`` already use that convention -- one line per
        agent in the given order.
        """
        # A generous makespan cap so PIBT2 never truncates a valid solution
        # on the small/medium grids used in the ablation.
        max_timestep = max(1000, graph.width * graph.height)
        max_comp_time_ms = max(1, int(round(time_limit_s * 1000)))

        lines = [
            f"map_file={map_name}",
            f"agents={len(agents)}",
            "seed=0",
            "random_problem=0",
            f"max_timestep={max_timestep}",
            f"max_comp_time={max_comp_time_ms}",
        ]
        for a in agents:
            lines.append(f"{a.start[0]},{a.start[1]},{a.goal[0]},{a.goal[1]}")
        return "\n".join(lines) + "\n"

    def _run_pibt(
        self, args: Sequence[str], tmpdir: Path, time_limit_s: float
    ) -> None:
        """Run the PIBT2 binary with ``cwd=tmpdir`` and a wall-clock timeout.

        The base :meth:`_run_subprocess` cannot be reused here because it
        does not accept a ``cwd`` (PIBT2's ``map_file`` resolution requires
        running inside ``tmpdir``). Error/timeout semantics mirror the base
        helper: :class:`SolverTimeoutError` on timeout, :class:`SolverError`
        on non-zero exit.
        """
        try:
            proc = subprocess.run(  # noqa: S603 - args constructed by trusted code
                list(args),
                check=False,
                capture_output=True,
                text=True,
                timeout=time_limit_s + 1.0,
                cwd=str(tmpdir),
            )
        except subprocess.TimeoutExpired as exc:
            raise SolverTimeoutError(
                f"{type(self).__name__}: subprocess exceeded "
                f"{time_limit_s:.2f} s wall-clock budget"
            ) from exc
        if proc.returncode != 0:
            raise SolverError(
                f"{type(self).__name__}: subprocess exited with code "
                f"{proc.returncode}; stderr=\n{proc.stderr.strip()}"
            )

    def _check_solved(self, solution_text: str) -> None:
        """Raise :class:`SolverError` when PIBT reports ``solved=0``.

        PIBT is incomplete: on failure it still writes an output file but
        with ``solved=0`` and no ``solution=`` block. Detect the explicit
        flag and fail loudly rather than handing empty text to the parser.
        """
        for line in solution_text.splitlines():
            if line.strip() == "solved=0":
                raise SolverError(
                    f"{type(self).__name__}: PIBT reported solved=0 "
                    "(no solution found within the time/timestep budget)"
                )

    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        """Parse PIBT's solution text into a :class:`Plan` (reuses LaCAM*).

        The ``solution=`` block format is identical to LaCAM*'s, so we
        delegate to the verified :meth:`LaCAMStarSolver._parse_output`
        rather than duplicating the per-timestep parser.
        """
        helper = LaCAMStarSolver(binary_path=self.binary_path)
        return helper._parse_output(stdout, agents)  # noqa: SLF001 - shared format
