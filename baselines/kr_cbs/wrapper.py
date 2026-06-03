"""Wrappers for k-Robust EECBS and k-Robust PBS.

Both binaries reuse the EECBS plain-text output format
(``Agent N: (x,y)->(x,y)->...``), so the parser delegates to
:class:`slackcertify.solvers.eecbs.EECBSSolver`.

Returned :class:`baselines._types.BaselinePlan` carry the
``solver_used`` string ``"kr_eecbs_k=K_w=W"`` (or ``kr_pbs_k=K_w=W``)
so the analysis layer can distinguish them.
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

__all__ = ["KRobustCBSBaseline", "KRobustPBSBaseline"]


class _KRobustBase(BinarySolverBase):
    """Shared scaffolding for kR-EECBS and kR-PBS wrappers."""

    NAME: ClassVar[str] = "kr_unspecified"
    MARKER: ClassVar[str] = "kr_unspecified"

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
        k_raw = kwargs.get("k", 1)
        if not isinstance(k_raw, int):
            raise TypeError(f"k must be an int, got {type(k_raw).__name__}")
        sub_raw = kwargs.get("suboptimality", 1.2)
        if not isinstance(sub_raw, (int, float)):
            raise TypeError(f"suboptimality must be a number, got {type(sub_raw).__name__}")
        suboptimality = float(sub_raw)
        output_file = tmpdir / f"{self.NAME}_output.txt"
        return [
            str(self.binary_path),
            "-m",
            str(map_path),
            "-a",
            str(scen_path),
            "-k",
            str(n_agents),
            "--k-robust",
            str(k_raw),
            "--suboptimality",
            f"{suboptimality:g}",
            "-o",
            str(output_file),
            "-t",
            f"{time_limit_s:g}",
        ]

    def _parse_output(self, stdout: str, agents: Sequence[Agent]) -> Plan:
        # k-Robust EECBS / PBS reuse EECBS's "Agent N: (x,y)->..." format.
        return EECBSSolver(binary_path=self.binary_path)._parse_output(  # noqa: SLF001
            stdout, agents
        )


class KRobustCBSBaseline(_KRobustBase):
    """k-Robust EECBS wrapper (Atzmon JAIR 2020 / Chen AAAI 2021)."""

    NAME: ClassVar[str] = "kr_eecbs"
    MARKER: ClassVar[str] = "kr_eecbs"

    def solve(  # type: ignore[override]
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        k: int,
        time_limit_s: float = 60.0,
        suboptimality: float = 1.2,
        **kwargs: object,
    ) -> BaselinePlan:
        plan = super().solve(
            graph,
            agents,
            time_limit_s,
            k=k,
            suboptimality=suboptimality,
            **kwargs,
        )
        return BaselinePlan(
            plan=plan,
            solver_used=f"{self.MARKER}_k={k}_w={suboptimality:g}",
        )


class KRobustPBSBaseline(_KRobustBase):
    """k-Robust PBS wrapper (Chen et al., AAAI 2021)."""

    NAME: ClassVar[str] = "kr_pbs"
    MARKER: ClassVar[str] = "kr_pbs"

    def solve(  # type: ignore[override]
        self,
        graph: GridGraph,
        agents: Sequence[Agent],
        k: int,
        time_limit_s: float = 60.0,
        suboptimality: float = 1.2,
        **kwargs: object,
    ) -> BaselinePlan:
        plan = super().solve(
            graph,
            agents,
            time_limit_s,
            k=k,
            suboptimality=suboptimality,
            **kwargs,
        )
        return BaselinePlan(
            plan=plan,
            solver_used=f"{self.MARKER}_k={k}_w={suboptimality:g}",
        )
