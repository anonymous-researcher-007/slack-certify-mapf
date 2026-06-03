"""Wrapper for p-Robust CBS (Atzmon et al., ICAPS 2020).

The published scaling limit is ~8 agents on 8x8 empty grids; the wrapper
therefore raises :class:`SolverTimeoutError` gracefully and callers should
treat timeouts as expected for larger instances.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from baselines._paths import external_bin
from baselines._types import BaselinePlan
from slackcertify.core.graph import GridGraph
from slackcertify.core.plan import Agent, Plan
from slackcertify.solvers.base import BinarySolverBase
from slackcertify.solvers.eecbs import EECBSSolver

__all__ = ["PRobustCBSBaseline"]


class PRobustCBSBaseline(BinarySolverBase):
    """p-Robust CBS wrapper. Output reuses the EECBS plain-text path format."""

    NAME: ClassVar[str] = "pr_cbs"
    MARKER: ClassVar[str] = "pr_cbs"

    @classmethod
    def _default_binary_path(cls) -> Path:
        return external_bin(cls.NAME)

    def _build_args(
        self,
        map_path: Path,
        scen_path: Path,
        n_agents: int,
        time_limit_s: float,
        tmpdir: Path,
        **kwargs: object,
    ) -> list[str]:
        p_robust_raw = kwargs.get("p_robust")
        p_d_raw = kwargs.get("p_d")
        if not isinstance(p_robust_raw, (int, float)):
            raise TypeError(f"p_robust must be a float in [0, 1], got {p_robust_raw!r}")
        if not isinstance(p_d_raw, (int, float)):
            raise TypeError(f"p_d must be a float in [0, 1), got {p_d_raw!r}")
        output_file = tmpdir / "pr_cbs_output.txt"
        return [
            str(self.binary_path),
            "-m",
            str(map_path),
            "-a",
            str(scen_path),
            "-k",
            str(n_agents),
            "--p-robust",
            f"{float(p_robust_raw):g}",
            "--delay-prob",
            f"{float(p_d_raw):g}",
            "-o",
            str(output_file),
            "-t",
            f"{time_limit_s:g}",
        ]

    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        return EECBSSolver(binary_path=self.binary_path)._parse_output(  # noqa: SLF001
            stdout, agents
        )

    def solve(  # type: ignore[override]
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        p_robust: float,
        p_d: float,
        time_limit_s: float = 60.0,
        **kwargs: object,
    ) -> BaselinePlan:
        if not (0.0 <= p_robust <= 1.0):
            raise ValueError(f"p_robust must lie in [0, 1], got {p_robust}")
        if not (0.0 <= p_d < 1.0):
            raise ValueError(f"p_d must lie in [0, 1), got {p_d}")
        plan = super().solve(graph, agents, time_limit_s, p_robust=p_robust, p_d=p_d, **kwargs)
        return BaselinePlan(
            plan=plan,
            solver_used=f"{self.MARKER}_p={p_robust:g}_pd={p_d:g}",
        )
