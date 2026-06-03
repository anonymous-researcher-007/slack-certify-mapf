"""Kottinger CBS-delay wrapper (Kottinger et al. 2024, arXiv:2307.11252).

Real binary wrapper for the upstream
`aria-systems-group/Delay-Robust-MAPF` solver. The wrapper composes
the subprocess machinery from :class:`BinarySolverBase` (rather than
inheriting it) because the public API is :meth:`solve_offline`
``(plan, delta, time_limit_s)`` — a plan-in/plan-out post-processor
— not the canonical ``(graph, agents, time_limit_s)`` MAPF
front-end signature.

When the upstream binary is missing, the wrapper **raises**
:class:`SolverNotFoundError` rather than falling back to a stub
implementation. The §V runners catch this exception and record
``status="binary_missing"`` per the established convention; the
analysis pipeline filters such rows out of paper-grade plots and
tables. (Earlier versions of this module shipped a silent fallback
that delegated to ``slack_certify`` — see the D3 audit-resolution
discussion in :mod:`baselines.kottinger._reimpl` for the rationale
behind removing it.)

Two modes:

* :meth:`solve_offline(plan, delta, time_limit_s)` — primary §V baseline.
  Worst-case offline solve so the proactive-vs-reactive axis is the
  sole differentiator from Slack-Certify.
* :meth:`solve_online(plan, observed_delay, time_limit_s)` — reactive
  single-event solve, included for completeness; not on the §V
  critical path.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

from baselines._paths import external_bin
from baselines._types import BaselinePlan
from baselines.kottinger import _reimpl
from slackcertify.core.plan import Plan
from slackcertify.io.plan_io import load_plan, save_plan
from slackcertify.solvers.base import (
    SolverError,
    SolverNotFoundError,
    SolverTimeoutError,
)

__all__ = ["KottingerDelayBaseline"]

_INSTALL_HINT = (
    "Run scripts/install_baselines.sh to build the Kottinger binary "
    "from third_party/delay-introduction/."
)


class KottingerDelayBaseline:
    """Wrapper for the upstream Kottinger 2024 delay-introduction solver.

    Requires the binary at
    ``src/slackcertify/solvers/external_bin/kottinger``; raises
    :class:`SolverNotFoundError` if it is missing. The
    ``solver_used`` marker on the returned :class:`BaselinePlan` is
    always ``"<nominal>+kottinger_offline_delta=<delta>_binary"`` —
    there is no fallback provenance to track.
    """

    NAME: ClassVar[str] = "kottinger"
    MARKER: ClassVar[str] = "kottinger_offline_delta"

    def __init__(self, binary_path: str | Path | None = None) -> None:
        if binary_path is None:
            binary_path = external_bin(self.NAME)
        self._binary_path: Path = Path(binary_path)

    @property
    def binary_path(self) -> Path:
        """Filesystem path of the upstream Kottinger binary."""
        return self._binary_path

    @classmethod
    def _use_binary(cls, baseline: KottingerDelayBaseline) -> bool:
        """Return True iff the upstream binary exists and should be used."""
        return baseline._binary_path.exists()

    def solve_offline(self, plan: Plan, delta: int, time_limit_s: float = 60.0) -> BaselinePlan:
        """Run Kottinger in offline mode against worst-case ``delta``.

        Routes through the upstream binary. Raises
        :class:`SolverNotFoundError` if the binary is missing (no
        silent fallback — see the module docstring and
        :mod:`baselines.kottinger._reimpl` for the D3 rationale).
        """
        if delta < 0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if not KottingerDelayBaseline._use_binary(self):
            # Delegate to the fail-loud stub so the error message
            # (binary path + install hint + upstream URL) lives in
            # exactly one place. The stub always raises.
            _reimpl.kottinger_offline_solve(plan, delta=delta, time_limit_s=time_limit_s)
        pi_prime = self._solve_offline_binary(plan, delta, time_limit_s)
        return BaselinePlan(
            plan=pi_prime,
            solver_used=f"<nominal>+{self.MARKER}={delta}_binary",
        )

    def solve_online(
        self, plan: Plan, observed_delay: object, time_limit_s: float = 1.0
    ) -> BaselinePlan:
        """Reactive single-event solve. Not on the §V critical path."""
        pi_prime = _reimpl.kottinger_online_solve(
            plan, observed_delay=observed_delay, time_limit_s=time_limit_s
        )
        return BaselinePlan(plan=pi_prime, solver_used="<nominal>+kottinger_online")

    # ---------------------------------------------------- binary path internals

    def _solve_offline_binary(self, plan: Plan, delta: int, time_limit_s: float) -> Plan:
        """Invoke the upstream binary on ``plan`` and read back the result.

        The CLI interface here uses ``TODO_VERIFY`` placeholders for
        flag names and the output-parsing strategy; the maintainer will confirm
        against the upstream README during the first real build
        attempt. Raises :class:`SolverNotFoundError` if the binary is
        missing (so the runner can surface ``status="binary_missing"``).
        """
        if not self._binary_path.exists():
            raise SolverNotFoundError(
                f"{type(self).__name__}: Kottinger binary not found at "
                f"{self._binary_path}. {_INSTALL_HINT}"
            )
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            input_path = tmpdir / "plan.json"
            output_path = tmpdir / "plan_delayed.json"
            save_plan(plan, input_path)
            # TODO_VERIFY: confirm the upstream CLI flag names. The
            # placeholder set follows the MovingAI/k-Robust pattern
            # (``-m``/``-a``) extended with ``--delta`` and
            # ``--time-limit``; the actual flags may use
            # ``--input/--output`` like BTPG, or a single positional
            # JSON path. Update once the maintainer confirms.
            args = [
                str(self._binary_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--delta",
                str(delta),
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
                    f"Kottinger binary exceeded {time_limit_s:.2f} s wall-clock budget"
                ) from exc
            if proc.returncode != 0:
                raise SolverError(
                    f"Kottinger binary exited with code {proc.returncode}; "
                    f"stderr=\n{proc.stderr.strip()}"
                )
            # TODO_VERIFY: confirm the upstream produces a JSON plan in
            # the same schema slackcertify.io.plan_io expects. If the
            # output format differs (e.g. plain-text per-agent
            # vertex lists like EECBS), parse it explicitly here
            # instead of using load_plan.
            if not output_path.exists():
                raise SolverError(
                    f"Kottinger binary did not produce {output_path} "
                    f"(stdout truncated to 1 KiB):\n{proc.stdout[:1024]}"
                )
            return load_plan(output_path)
