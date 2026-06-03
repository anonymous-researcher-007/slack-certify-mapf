"""Command-line entry points (Typer)."""

from __future__ import annotations

from slackcertify.cli import certify, pipeline, simulate
from slackcertify.cli.__main__ import app, main

__all__ = ["app", "certify", "main", "pipeline", "simulate"]
