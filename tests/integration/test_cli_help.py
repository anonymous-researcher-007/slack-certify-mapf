"""Integration tests: every CLI sub-command shows ``--help`` cleanly."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from slackcertify.cli.__main__ import app

runner = CliRunner()


@pytest.mark.integration
def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "slackcertify" in result.output.lower()
    assert "certify" in result.output
    assert "simulate" in result.output
    assert "pipeline" in result.output


@pytest.mark.integration
def test_root_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "slackcertify" in result.output


@pytest.mark.integration
def test_certify_help() -> None:
    result = runner.invoke(app, ["certify", "--help"])
    assert result.exit_code == 0, result.output
    for token in ("--plan", "--mode", "--delta", "--p-d", "--epsilon", "--out"):
        assert token in result.output, f"missing flag {token} in:\n{result.output}"


@pytest.mark.integration
def test_simulate_help() -> None:
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0, result.output
    for token in (
        "--plan",
        "--delay-model",
        "--delta",
        "--p-d",
        "--rollouts",
        "--seed",
        "--out",
    ):
        assert token in result.output, f"missing flag {token} in:\n{result.output}"


@pytest.mark.integration
def test_pipeline_help() -> None:
    result = runner.invoke(app, ["pipeline", "--help"])
    assert result.exit_code == 0, result.output
    for token in (
        "--map",
        "--scen",
        "--n-agents",
        "--solver",
        "--mode",
        "--delta",
        "--p-d",
        "--rollouts",
        "--out-dir",
        "--seed",
    ):
        assert token in result.output, f"missing flag {token} in:\n{result.output}"


@pytest.mark.integration
def test_unknown_subcommand_errors() -> None:
    result = runner.invoke(app, ["nope"])
    assert result.exit_code != 0
