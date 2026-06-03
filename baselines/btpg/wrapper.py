"""Wrapper for BTPG-max (Su et al., AAAI 2024 / arXiv:2508.04849, 2025).

BTPG is *post-hoc*: it relaxes Type-2 TPG edges of an existing plan to
maximise execution slack. The wrapper therefore exposes
:meth:`post_process(plan, time_limit_s)` rather than a ``solve()``
method, and composes — rather than inherits — the subprocess machinery
from :class:`slackcertify.solvers.base.BinarySolverBase`.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

from baselines._paths import external_bin
from baselines._types import BaselinePlan
from slackcertify.core.plan import Plan
from slackcertify.io.plan_io import load_plan, save_plan
from slackcertify.solvers.base import (
    SolverError,
    SolverNotFoundError,
    SolverTimeoutError,
)

__all__ = ["BTPGMaxBaseline"]

_INSTALL_HINT = "Run scripts/install_baselines.sh to build the BTPG binary."


class BTPGMaxBaseline:
    """Post-hoc TPG relaxation via the BTPG-max upstream binary."""

    NAME: ClassVar[str] = "btpg"
    MARKER: ClassVar[str] = "btpg_max"

    def __init__(self, binary_path: str | Path | None = None) -> None:
        if binary_path is None:
            binary_path = external_bin(self.NAME)
        self._binary_path: Path = Path(binary_path)

    @property
    def binary_path(self) -> Path:
        """Path to the BTPG binary."""
        return self._binary_path

    def _check_binary_exists(self) -> None:
        if not self._binary_path.exists():
            raise SolverNotFoundError(
                f"{type(self).__name__}: BTPG binary not found at "
                f"{self._binary_path}. {_INSTALL_HINT}"
            )

    def post_process(self, plan: Plan, time_limit_s: float = 60.0) -> BaselinePlan:
        """Run BTPG-max on ``plan`` and return the relaxed :class:`BaselinePlan`.

        Writes the nominal plan as JSON, invokes the binary with
        ``--input/--output/--mode max/--time-limit``, and reads back the
        relaxed plan from the output JSON. The returned
        :class:`BaselinePlan` is tagged ``"<nominal>+btpg_max"``.
        """
        self._check_binary_exists()
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            input_path = tmpdir / "plan.json"
            output_path = tmpdir / "relaxed.json"
            save_plan(plan, input_path)
            args = [
                str(self._binary_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--mode",
                "max",
                "--time-limit",
                f"{time_limit_s:g}",
            ]
            try:
                proc = subprocess.run(  # noqa: S603 - argv is constructed locally
                    args,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=time_limit_s + 5.0,
                )
            except subprocess.TimeoutExpired as exc:
                raise SolverTimeoutError(
                    f"BTPG-max exceeded {time_limit_s:.2f} s wall-clock budget"
                ) from exc
            if proc.returncode != 0:
                raise SolverError(
                    f"BTPG-max exited with code {proc.returncode}; "
                    f"stderr=\n{proc.stderr.strip()}"
                )
            if not output_path.exists():
                raise SolverError(
                    f"BTPG-max did not produce {output_path} (stdout truncated to 1 KiB):\n"
                    + proc.stdout[:1024]
                )
            relaxed = load_plan(output_path)
        return BaselinePlan(plan=relaxed, solver_used=f"<nominal>+{self.MARKER}")
