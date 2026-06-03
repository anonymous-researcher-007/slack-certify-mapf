"""``slackcertify pipeline`` — end-to-end map → solver → certify → rollout.

The solver step is currently a placeholder: solver wrappers will land in a
follow-up commit. Until then, the CLI exposes the same flag surface but
raises :class:`NotImplementedError` when invoked. The pure-Python
:func:`run` function exposed here mirrors the CLI flags and is what the
README quick-start example references.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from slackcertify.certify.certificate import Certificate
from slackcertify.core.plan import Plan
from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.io.certificate_io import save_certificate
from slackcertify.io.plan_io import save_plan
from slackcertify.io.rollout_io import save_rollout_results
from slackcertify.repair import slack_certify
from slackcertify.simulate.rollout import RolloutResult, monte_carlo_rollout

__all__ = ["PipelineResult", "pipeline_cmd", "run"]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Aggregate output of an end-to-end pipeline run."""

    plan: Plan
    certified_plan: Plan
    certificate: Certificate
    rollout: RolloutResult

    @property
    def sum_of_costs(self) -> int:
        """Sum-of-costs of the certified plan."""
        return self.certified_plan.sum_of_costs

    @property
    def makespan(self) -> int:
        """Makespan of the certified plan."""
        return self.certified_plan.makespan

    @property
    def rollout_success_rate(self) -> float:
        """Empirical success rate from the rollout study."""
        return self.rollout.success_rate


def run(
    map_path: str | Path,
    scen_path: str | Path,
    n_agents: int,
    solver: str,
    delay_model: dict[str, object],
    n_rollouts: int = 500,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> PipelineResult:
    """Run the end-to-end pipeline programmatically.

    Currently raises :class:`NotImplementedError` for the solver step;
    the function signature is finalised so callers (notebooks, the README
    quick-start, the CLI sub-command) can be written today and start
    working as soon as the solver wrappers land.
    """
    _ = (
        map_path,
        scen_path,
        n_agents,
        solver,
        delay_model,
        n_rollouts,
        seed,
        out_dir,
    )
    raise NotImplementedError(
        "slackcertify.cli.pipeline.run is a stub: solver wrappers (EECBS, "
        "LaCAM, PIBT) have not yet been implemented. Provide a precomputed "
        "plan via `slackcertify certify --plan PLAN.json` or import the "
        "library directly until the solvers land."
    )


def pipeline_cmd(
    map_path: Annotated[Path, typer.Option("--map", help="MovingAI .map file.")],
    scen_path: Annotated[Path, typer.Option("--scen", help="MovingAI .scen file.")],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Output directory; created if missing.")
    ],
    n_agents: Annotated[int, typer.Option("--n-agents", help="Number of agents.")] = 8,
    solver: Annotated[
        str,
        typer.Option(
            "--solver",
            help="MAPF solver: eecbs | lacam | pibt.",
            case_sensitive=False,
        ),
    ] = "eecbs",
    mode: Annotated[
        str,
        typer.Option("--mode", help="bounded | probabilistic", case_sensitive=False),
    ] = "bounded",
    delta: Annotated[int | None, typer.Option("--delta", help="Bounded-mode budget.")] = None,
    p_d: Annotated[
        float | None,
        typer.Option("--p-d", help="Per-step delay probability (probabilistic mode)."),
    ] = None,
    epsilon: Annotated[
        float | None,
        typer.Option("--epsilon", help="Union-bound risk budget (probabilistic mode)."),
    ] = None,
    rollouts: Annotated[
        int, typer.Option("--rollouts", help="Number of Monte-Carlo rollouts.")
    ] = 500,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed.")] = 0,
) -> None:
    """End-to-end MAPF + slack-certify + rollout pipeline.

    The solver step is not yet implemented; this command therefore
    raises :class:`NotImplementedError` after parsing its arguments.
    """
    _ = (
        map_path,
        scen_path,
        out_dir,
        n_agents,
        solver,
        mode,
        delta,
        p_d,
        epsilon,
        rollouts,
        seed,
    )
    raise typer.Exit(
        code=2,
    )  # pragma: no cover - the command intentionally aborts until solvers land


def _certify_and_rollout(
    plan: Plan,
    *,
    mode: str,
    delta: int | None,
    p_d: float | None,
    epsilon: float | None,
    rollouts: int,
    seed: int,
    out_dir: Path,
) -> PipelineResult:
    """Helper used once solver wrappers exist (kept for early integration tests)."""
    # ``mode`` is parsed from argv as ``str`` but validated against a
    # finite list earlier in the function, so the cast is sound at
    # this call site. Documenting it via :func:`typing.cast` lets
    # mypy strict pass without an unused-ignore comment.
    pi_prime, cert = slack_certify(
        plan,
        mode=cast(Literal["bounded", "probabilistic"], mode),
        delta=delta,
        p_d=p_d,
        epsilon=epsilon,
    )

    model: BoundedDelayModel | BernoulliDelayModel
    if mode == "bounded":
        assert delta is not None
        model = BoundedDelayModel(delta=delta)
    else:
        assert p_d is not None
        model = BernoulliDelayModel(p_d=p_d)

    rng = np.random.default_rng(seed)
    rollout = monte_carlo_rollout(pi_prime, model, K=rollouts, rng=rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_plan(pi_prime, out_dir / "plan.certified.json")
    save_certificate(cert, out_dir / "certificate.json")
    save_rollout_results(rollout, out_dir / "rollout.parquet")

    _print_summary(plan, pi_prime, cert, rollout, out_dir)
    return PipelineResult(plan=plan, certified_plan=pi_prime, certificate=cert, rollout=rollout)


def _print_summary(
    plan: Plan,
    certified: Plan,
    cert: Certificate,
    rollout: RolloutResult,
    out_dir: Path,
) -> None:
    """Print a rich-formatted pipeline summary (solver → certify → rollout) to stdout."""
    console = Console()
    table = Table(title="Pipeline result", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("agents", str(len(plan.agents)))
    table.add_row("input makespan / SoC", f"{plan.makespan} / {plan.sum_of_costs}")
    table.add_row("certified makespan / SoC", f"{certified.makespan} / {certified.sum_of_costs}")
    table.add_row("waits inserted", str(cert.total_wait_inserted))
    table.add_row("rollout success rate", f"{rollout.success_rate:.4f}")
    table.add_row("output dir", str(out_dir))
    console.print(table)
