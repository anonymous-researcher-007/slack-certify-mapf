"""``slackcertify simulate`` — Monte-Carlo rollout of a plan under a delay model."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from slackcertify.delay.bernoulli import BernoulliDelayModel
from slackcertify.delay.bounded import BoundedDelayModel
from slackcertify.delay.persistent import PersistentDelayModel
from slackcertify.io.plan_io import load_plan
from slackcertify.io.rollout_io import save_rollout_results
from slackcertify.simulate.rollout import DelayModel, RolloutResult, monte_carlo_rollout

__all__ = ["simulate_cmd"]


def simulate_cmd(
    plan: Annotated[Path, typer.Option("--plan", help="Plan JSON to roll out.")],
    out: Annotated[
        Path,
        typer.Option("--out", help="Parquet file to write rollout results to."),
    ],
    delay_model: Annotated[
        str,
        typer.Option(
            "--delay-model",
            help="One of bounded | bernoulli | persistent.",
            case_sensitive=False,
        ),
    ] = "bernoulli",
    delta: Annotated[
        int | None,
        typer.Option("--delta", help="Bounded-model budget."),
    ] = None,
    p_d: Annotated[
        float | None,
        typer.Option("--p-d", help="Per-step delay probability (Bernoulli / persistent trigger)."),
    ] = None,
    persistent_min_len: Annotated[
        int,
        typer.Option("--persistent-min-len", help="Minimum stall length (persistent)."),
    ] = 10,
    persistent_max_len: Annotated[
        int,
        typer.Option("--persistent-max-len", help="Maximum stall length (persistent)."),
    ] = 20,
    rollouts: Annotated[
        int, typer.Option("--rollouts", help="Number of Monte-Carlo rollouts.")
    ] = 500,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed.")] = 0,
    n_jobs: Annotated[
        int, typer.Option("--n-jobs", help="Joblib parallelism (-1 = all cores).")
    ] = 1,
) -> None:
    """Run ``--rollouts`` Monte-Carlo rollouts of ``--plan`` under the chosen delay model."""
    plan_obj = load_plan(plan)
    model = _build_delay_model(
        delay_model,
        delta=delta,
        p_d=p_d,
        min_len=persistent_min_len,
        max_len=persistent_max_len,
    )

    rng = np.random.default_rng(seed)
    result = monte_carlo_rollout(plan_obj, model, K=rollouts, rng=rng, n_jobs=n_jobs)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_rollout_results(result, out)
    _print_summary(result, out)


def _build_delay_model(
    name: str, *, delta: int | None, p_d: float | None, min_len: int, max_len: int
) -> DelayModel:
    """Construct a :class:`DelayModel` from the CLI's ``--delay-model`` choice."""
    n = name.lower()
    if n == "bounded":
        if delta is None:
            raise typer.BadParameter("--delta is required for delay-model=bounded")
        return BoundedDelayModel(delta=delta)
    if n == "bernoulli":
        if p_d is None:
            raise typer.BadParameter("--p-d is required for delay-model=bernoulli")
        return BernoulliDelayModel(p_d=p_d)
    if n == "persistent":
        if p_d is None:
            raise typer.BadParameter("--p-d is required for delay-model=persistent")
        return PersistentDelayModel(p_trigger=p_d, min_len=min_len, max_len=max_len)
    raise typer.BadParameter(
        f"delay-model must be one of bounded|bernoulli|persistent, got {name!r}"
    )


def _print_summary(result: RolloutResult, out: Path) -> None:
    """Print a rich-formatted Monte-Carlo rollout summary to stdout."""
    console = Console()
    lo, hi = result.wilson_ci_95
    table = Table(title="Monte-Carlo rollout result", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("rollouts", str(result.n_rollouts))
    table.add_row("successful", str(result.n_successful))
    table.add_row("success rate", f"{result.success_rate:.4f}")
    table.add_row("Wilson 95% CI", f"[{lo:.4f}, {hi:.4f}]")
    table.add_row("mean executed SoC", f"{result.mean_executed_soc:.3f}")
    table.add_row("mean executed makespan", f"{result.mean_executed_makespan:.3f}")
    table.add_row("written to", str(out))
    console.print(table)
