#!/usr/bin/env python3
"""Diagnose the local toolchain for slack-certify-mapf.

Walks a list of *required* and *optional* checks (Python version, command-line
tools, Python imports, vendored solver binaries) and prints a green check or
red cross for each. Exits ``1`` if any required check fails, ``0`` otherwise.

Run from anywhere::

    python3 scripts/check_environment.py
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_BIN_DIR = REPO_ROOT / "src" / "slackcertify" / "solvers" / "external_bin"

_OK = "[green]✓[/green]"
_FAIL = "[red]✗[/red]"
_WARN = "[yellow]?[/yellow]"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One row in the environment-check report."""

    name: str
    status: bool
    detail: str
    required: bool


# ----------------------------------------------------------------- helpers


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(  # noqa: S603 - args from a hard-coded list
            cmd, check=False, capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    parts = tuple(int(g) for g in match.groups() if g is not None)
    return parts


def _check_python(min_version: tuple[int, int]) -> CheckResult:
    actual = sys.version_info[:2]
    ok = actual >= min_version
    return CheckResult(
        name=f"python >= {min_version[0]}.{min_version[1]}",
        status=ok,
        detail=f"detected {actual[0]}.{actual[1]}.{sys.version_info.micro}",
        required=True,
    )


def _check_cmd(
    name: str,
    *,
    version_cmd: list[str] | None = None,
    min_version: tuple[int, ...] | None = None,
    required: bool = True,
) -> CheckResult:
    found = shutil.which(name)
    if found is None:
        return CheckResult(name=name, status=False, detail="not on PATH", required=required)
    if version_cmd is None:
        return CheckResult(name=name, status=True, detail=f"at {found}", required=required)
    code, out = _run(version_cmd)
    if code != 0:
        return CheckResult(
            name=name,
            status=False,
            detail=f"version command failed: {' '.join(version_cmd)}",
            required=required,
        )
    parsed = _parse_version(out.splitlines()[0] if out else "")
    if min_version is not None and parsed is not None and parsed < min_version:
        wanted = ".".join(str(v) for v in min_version)
        got = ".".join(str(v) for v in parsed)
        return CheckResult(
            name=f"{name} >= {wanted}",
            status=False,
            detail=f"found {got} at {found}",
            required=required,
        )
    detail = "ok"
    if parsed is not None:
        detail = ".".join(str(v) for v in parsed)
    return CheckResult(name=name, status=True, detail=detail, required=required)


def _check_module(modname: str, *, required: bool = False) -> CheckResult:
    spec = importlib.util.find_spec(modname)
    return CheckResult(
        name=f"python: {modname}",
        status=spec is not None,
        detail="importable" if spec is not None else "not installed",
        required=required,
    )


def _check_solver_binary(name: str) -> CheckResult:
    path = EXTERNAL_BIN_DIR / name
    if path.exists():
        size = path.stat().st_size
        return CheckResult(
            name=f"solver binary: {name}",
            status=True,
            detail=f"{path} ({size} bytes)",
            required=False,
        )
    return CheckResult(
        name=f"solver binary: {name}",
        status=False,
        detail=f"missing at {path}; run scripts/install_baselines.sh",
        required=False,
    )


# ----------------------------------------------------------------- main


def collect_checks() -> list[CheckResult]:
    out: list[CheckResult] = []

    # Required ----------------------------------------------------------------
    out.append(_check_python((3, 10)))
    out.append(_check_cmd("pip", version_cmd=["pip", "--version"]))
    out.append(_check_cmd("git", version_cmd=["git", "--version"]))
    out.append(_check_cmd("cmake", version_cmd=["cmake", "--version"], min_version=(3, 16)))
    out.append(_check_cmd("g++", version_cmd=["g++", "--version"], min_version=(11,)))

    # Optional Python deps for ILP / parallel / Parquet -----------------------
    for mod in ("gurobipy", "highspy", "pulp"):
        out.append(_check_module(mod, required=False))
    out.append(_check_module("joblib", required=False))
    out.append(_check_module("pyarrow", required=False))

    # Optional solver binaries ------------------------------------------------
    for name in ("eecbs", "lacam", "lacam_star", "pibt", "btpg", "kottinger"):
        out.append(_check_solver_binary(name))

    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-binaries",
        action="store_true",
        help=(
            "Promote every solver-binary check to required, so the "
            "script exits non-zero when any baseline binary is missing. "
            "Used by scripts/repro_paper.sh to gate the full §V sweep."
        ),
    )
    args = parser.parse_args()

    console = Console()
    results = collect_checks()
    if args.require_binaries:
        promoted: list[CheckResult] = []
        for r in results:
            if r.name.startswith("solver binary:"):
                promoted.append(
                    CheckResult(
                        name=r.name,
                        status=r.status,
                        detail=r.detail,
                        required=True,
                    )
                )
            else:
                promoted.append(r)
        results = promoted

    table = Table(title="slack-certify-mapf environment check", show_lines=False)
    table.add_column("status", justify="center", width=4)
    table.add_column("required", justify="center", width=8)
    table.add_column("check", style="bold")
    table.add_column("detail")

    n_required_failed = 0
    for r in results:
        marker = _OK if r.status else (_FAIL if r.required else _WARN)
        if r.required and not r.status:
            n_required_failed += 1
        table.add_row(marker, "yes" if r.required else "no", r.name, r.detail)

    console.print(table)
    if n_required_failed:
        console.print(f"[red]✗ {n_required_failed} required check(s) failed.[/red]")
        return 1
    console.print("[green]✓ all required checks passed.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
