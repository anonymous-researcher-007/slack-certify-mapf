"""Resolve the ``slackcertify/solvers/external_bin`` directory from a baseline."""

from __future__ import annotations

from pathlib import Path

__all__ = ["external_bin"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTERNAL_BIN_DIR = _REPO_ROOT / "src" / "slackcertify" / "solvers" / "external_bin"


def external_bin(name: str) -> Path:
    """Return the canonical path of solver binary ``name`` under ``external_bin/``."""
    return _EXTERNAL_BIN_DIR / name
