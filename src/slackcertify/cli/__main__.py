"""Typer entry point for the ``slackcertify`` CLI.

The console script ``slackcertify = slackcertify.cli.__main__:app`` is
declared in ``pyproject.toml``; running ``slackcertify --help`` prints
the table of sub-commands.
"""

from __future__ import annotations

import typer

from slackcertify._version import __version__
from slackcertify.cli.certify import certify_cmd
from slackcertify.cli.pipeline import pipeline_cmd
from slackcertify.cli.simulate import simulate_cmd

__all__ = ["app", "main"]

app = typer.Typer(
    name="slackcertify",
    help=(
        "Slack-Certified One-Shot MAPF: proactive, solver-agnostic wait "
        "insertion for bounded and probabilistic delay tolerance."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)


def _version_callback(value: bool) -> None:
    """Print the package version and exit when ``--version`` fires."""
    if value:
        typer.echo(f"slackcertify {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def _root(
    version: bool = typer.Option(  # noqa: ARG001 - consumed by the eager callback
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the package version and exit.",
    ),
) -> None:
    """Show ``--version`` or hand off to a sub-command."""


app.command(name="certify", help="Slack-certify a plan.")(certify_cmd)
app.command(name="simulate", help="Monte-Carlo rollout of a plan under delays.")(simulate_cmd)
app.command(
    name="pipeline",
    help="End-to-end map -> solver -> certify -> rollout (solver TODO).",
)(pipeline_cmd)


def main() -> None:  # pragma: no cover - thin alias used by some packagers
    """Entry-point alias used by ``console_scripts`` packagers."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
