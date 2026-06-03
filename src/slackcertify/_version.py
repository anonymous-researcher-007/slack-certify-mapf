"""Single source of truth for the package version.

This file is intentionally tiny so that build backends and runtime callers can
read it without importing the rest of the package.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.1.0"
