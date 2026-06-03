"""Nox sessions for slack-certify-mapf.

Run `nox --list` to see all sessions, or `nox -s <name>` to run one.
"""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "typecheck", "tests"]
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]
PACKAGE = "slackcertify"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the pytest suite on the requested Python versions."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run ruff and black --check."""
    session.install("ruff>=0.4", "black>=24.3")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("black", "--check", ".")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run mypy --strict on the slackcertify package."""
    session.install("-e", ".[dev]")
    session.run("mypy", "--strict", f"src/{PACKAGE}")


@nox.session(python="3.11")
def docs(session: nox.Session) -> None:
    """Build the mkdocs documentation site."""
    session.install("-e", ".[dev,docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="3.11")
def smoke(session: nox.Session) -> None:
    """Run the reproducibility smoke test."""
    session.install("-e", ".[dev]")
    session.run("bash", "scripts/repro_smoke.sh", external=True)
