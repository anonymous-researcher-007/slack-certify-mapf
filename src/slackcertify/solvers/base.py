"""Solver protocol and shared subprocess scaffolding for binary MAPF solvers.

Every wrapper in this package implements the :class:`SolverProtocol` interface
— a single :meth:`solve` method that takes a graph, a list of agents, and a
wall-clock time limit, and returns a :class:`~slackcertify.core.plan.Plan`.

The :class:`BinarySolverBase` ABC provides everything a "spawn an external
binary, parse its stdout" wrapper needs: pre-flight binary existence check,
MovingAI ``.map`` / ``.scen`` writer, ``subprocess.run`` driver with timeout
handling, and the abstract :meth:`_build_args` / :meth:`_parse_output` hooks.

Concrete subclasses only have to override those two abstract methods plus
optionally :attr:`NAME` (used to derive the default binary path under
``slackcertify/solvers/external_bin/``).
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan

__all__ = [
    "BinarySolverBase",
    "SolverError",
    "SolverNotFoundError",
    "SolverOutputParseError",
    "SolverProtocol",
    "SolverTimeoutError",
]

_INSTALL_HINT = "Run scripts/install_baselines.sh to build the solver binaries."


class SolverError(RuntimeError):
    """Base class for solver-related failures."""


class SolverNotFoundError(SolverError, FileNotFoundError):
    """Solver binary is missing at the expected path."""


class SolverTimeoutError(SolverError, TimeoutError):
    """Solver subprocess exceeded the requested wall-clock time limit."""


class SolverOutputParseError(SolverError):
    """Solver stdout could not be parsed into a :class:`Plan`."""


@runtime_checkable
class SolverProtocol(Protocol):
    """Structural protocol every solver wrapper satisfies."""

    def solve(  # noqa: D401 - protocol declaration
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        time_limit_s: float,
        **kwargs: object,
    ) -> Plan:
        """Solve the MAPF instance and return a :class:`Plan`."""
        ...


class BinarySolverBase(ABC):
    """ABC for solvers that shell out to an external binary.

    Subclasses must override :meth:`_build_args` and :meth:`_parse_output`,
    and typically set the class attribute :attr:`NAME` so the default
    binary path is derived as
    ``slackcertify/solvers/external_bin/<NAME>``.

    The concrete :meth:`solve` method handles binary existence checks,
    temp-directory plumbing, MovingAI input writing, subprocess
    invocation with timeout, and stdout parsing.
    """

    NAME: ClassVar[str] = "binary"

    def __init__(self, binary_path: str | Path | None = None) -> None:
        """Store the binary path; defaults to ``external_bin/<NAME>``."""
        if binary_path is None:
            binary_path = self._default_binary_path()
        self._binary_path: Path = Path(binary_path)

    # ------------------------------------------------------------------ paths

    @classmethod
    def _default_binary_path(cls) -> Path:
        """Return the canonical install path: ``external_bin/<NAME>``."""
        return Path(__file__).parent / "external_bin" / cls.NAME

    @property
    def binary_path(self) -> Path:
        """Filesystem path of the solver binary."""
        return self._binary_path

    def _check_binary_exists(self) -> None:
        """Raise :class:`SolverNotFoundError` if the binary is missing."""
        if not self._binary_path.exists():
            raise SolverNotFoundError(
                f"{type(self).__name__}: solver binary not found at "
                f"{self._binary_path}. {_INSTALL_HINT}"
            )

    # ------------------------------------------------------------- IO helpers

    def _write_movingai_input(
        self, graph: GridGraph, agents: Sequence[Agent], tmpdir: Path
    ) -> tuple[Path, Path]:
        """Write a MovingAI ``.map`` and ``.scen`` for the given instance.

        Returns the ``(map_path, scen_path)`` pair, both inside ``tmpdir``.

        The ``.map`` uses the standard "type octile" header; cells in
        ``graph.obstacles`` become ``@``, all other cells become ``.``.
        The ``.scen`` file emits one bucket-0 line per agent with start
        and goal coordinates and a placeholder optimal length of ``0``.
        """
        map_path = tmpdir / "instance.map"
        scen_path = tmpdir / "instance.scen"

        map_lines: list[str] = [
            "type octile",
            f"height {graph.height}",
            f"width {graph.width}",
            "map",
        ]
        for y in range(graph.height):
            row_chars: list[str] = []
            for x in range(graph.width):
                row_chars.append("@" if (x, y) in graph.obstacles else ".")
            map_lines.append("".join(row_chars))
        map_path.write_text("\n".join(map_lines) + "\n", encoding="utf-8")

        scen_lines: list[str] = ["version 1"]
        for a in agents:
            scen_lines.append(
                "\t".join(
                    str(field)
                    for field in (
                        0,
                        map_path.name,
                        graph.width,
                        graph.height,
                        a.start[0],
                        a.start[1],
                        a.goal[0],
                        a.goal[1],
                        0,
                    )
                )
            )
        scen_path.write_text("\n".join(scen_lines) + "\n", encoding="utf-8")
        return map_path, scen_path

    def _run_subprocess(self, args: Sequence[str], time_limit_s: float) -> tuple[str, str]:
        """Run ``args`` with a hard wall-clock timeout; return ``(stdout, stderr)``.

        Adds a one-second buffer to the timeout so the solver can clean
        up its own state before we kill it from the outside.

        Raises
        ------
        SolverTimeoutError
            On wall-clock timeout.
        SolverError
            On non-zero return code.
        """
        try:
            proc = subprocess.run(  # noqa: S603 - args are constructed by trusted code
                list(args),
                check=False,
                capture_output=True,
                text=True,
                timeout=time_limit_s + 1.0,
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
        return proc.stdout, proc.stderr

    # --------------------------------------------------------------- abstract

    @abstractmethod
    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        """Return the argv (including the binary) for the subprocess call."""

    @abstractmethod
    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        """Parse the solver's stdout into a :class:`Plan`."""

    # --------------------------------------------------------------- public

    def solve(
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        time_limit_s: float,
        **kwargs: object,
    ) -> Plan:
        """Solve the MAPF instance by shelling out to the wrapped binary.

        Parameters
        ----------
        graph
            The environment.
        agents
            The agents to plan for.
        time_limit_s
            Wall-clock budget in seconds. Subclasses are free to pass
            this through to the binary's own time-limit flag and to the
            subprocess timeout.
        **kwargs
            Solver-specific options (forwarded to :meth:`_build_args`).
        """
        self._check_binary_exists()
        import tempfile

        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            map_path, scen_path = self._write_movingai_input(graph, agents, tmpdir)
            args = self._build_args(
                map_path=map_path,
                scen_path=scen_path,
                n_agents=len(agents),
                time_limit_s=time_limit_s,
                tmpdir=tmpdir,
                **kwargs,
            )
            stdout, _ = self._run_subprocess(args, time_limit_s=time_limit_s)
            return self._parse_output(stdout, agents)
