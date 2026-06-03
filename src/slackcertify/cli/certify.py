"""``slackcertify certify`` — slack-certify a saved plan and emit a certificate."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from slackcertify.certify.certificate import Certificate
from slackcertify.core.plan import Plan
from slackcertify.io.certificate_io import save_certificate
from slackcertify.io.plan_io import load_plan, save_plan
from slackcertify.repair import slack_certify

__all__ = ["certify_cmd"]


def certify_cmd(
    plan: Annotated[Path, typer.Option("--plan", help="Input plan JSON.")],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory; will be created if it does not exist.",
        ),
    ],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Certification mode.", case_sensitive=False),
    ] = "bounded",
    delta: Annotated[
        int | None,
        typer.Option("--delta", help="Bounded-mode budget (Δ); required when mode=bounded."),
    ] = None,
    p_d: Annotated[
        float | None,
        typer.Option("--p-d", help="Per-step delay probability (mode=probabilistic)."),
    ] = None,
    epsilon: Annotated[
        float | None,
        typer.Option("--epsilon", help="Union-bound risk budget (mode=probabilistic)."),
    ] = None,
    ordering: Annotated[
        str,
        typer.Option(
            "--ordering",
            help="Conflict ordering: topological | risk | random.",
            case_sensitive=False,
        ),
    ] = "topological",
    budget_alloc: Annotated[
        str,
        typer.Option(
            "--budget-alloc",
            help="Per-conflict budget allocation: uniform | risk_proportional.",
            case_sensitive=False,
        ),
    ] = "uniform",
) -> None:
    """Slack-certify a plan and write the certified plan + certificate.

    Two artefacts are emitted under ``--out``:

    * ``plan.certified.json`` — the certified plan;
    * ``certificate.json`` — the matching machine-checkable certificate.
    """
    if mode not in ("bounded", "probabilistic"):
        raise typer.BadParameter(f"mode must be 'bounded' or 'probabilistic', got {mode!r}")
    if ordering not in ("topological", "risk", "random"):
        raise typer.BadParameter(
            f"ordering must be one of topological|risk|random, got {ordering!r}"
        )
    if budget_alloc not in ("uniform", "risk_proportional"):
        raise typer.BadParameter(
            f"budget-alloc must be uniform|risk_proportional, got {budget_alloc!r}"
        )

    plan_obj = load_plan(plan)
    pi_prime, cert = slack_certify(
        plan_obj,
        mode=mode,  # type: ignore[arg-type]
        delta=delta,
        p_d=p_d,
        epsilon=epsilon,
        ordering=ordering,  # type: ignore[arg-type]
        budget_alloc=budget_alloc,  # type: ignore[arg-type]
    )

    out.mkdir(parents=True, exist_ok=True)
    plan_out = out / "plan.certified.json"
    cert_out = out / "certificate.json"
    save_plan(pi_prime, plan_out)
    save_certificate(cert, cert_out)

    _print_summary(pi_prime, cert, plan_out, cert_out)


def _print_summary(pi_prime: Plan, cert: Certificate, plan_path: Path, cert_path: Path) -> None:
    """Print a rich-formatted summary of the certified plan to stdout."""
    console = Console()
    table = Table(title="Slack-Certify result", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("mode", cert.mode)
    if cert.delta is not None:
        table.add_row("delta", str(cert.delta))
    if cert.p_d is not None:
        table.add_row("p_d", f"{cert.p_d:.6g}")
    if cert.epsilon is not None:
        table.add_row("epsilon", f"{cert.epsilon:.6g}")
    table.add_row("agents", str(len(pi_prime.agents)))
    table.add_row("makespan", str(pi_prime.makespan))
    table.add_row("sum_of_costs", str(pi_prime.sum_of_costs))
    table.add_row("waits inserted", str(cert.total_wait_inserted))
    table.add_row("plan written to", str(plan_path))
    table.add_row("cert written to", str(cert_path))
    console.print(table)
