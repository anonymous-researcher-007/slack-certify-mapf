"""Input-validation tests for analysis.statistics.bootstrap.

These tests exist to exercise the ``raise ValueError`` paths that
the happy-path tests in :mod:`tests.unit.test_statistics` don't
hit. Bumps coverage on ``analysis/statistics/bootstrap.py`` above
the 80 % per-file threshold; supersedes the audit's D4 deferred
item.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis.statistics.bootstrap import (
    bootstrap_quantile,
    paired_bootstrap_mean_diff,
)


@pytest.mark.unit
def test_paired_bootstrap_raises_on_length_mismatch() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])  # different length
    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_mean_diff(a, b)


@pytest.mark.unit
def test_paired_bootstrap_raises_on_empty_input() -> None:
    empty = np.array([], dtype=float)
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_mean_diff(empty, empty)


@pytest.mark.unit
def test_paired_bootstrap_raises_on_invalid_alpha() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha must lie in"):
        paired_bootstrap_mean_diff(a, b, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must lie in"):
        paired_bootstrap_mean_diff(a, b, alpha=1.0)


@pytest.mark.unit
def test_paired_bootstrap_raises_on_invalid_n_bootstrap() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        paired_bootstrap_mean_diff(a, b, n_bootstrap=0)


@pytest.mark.unit
def test_bootstrap_quantile_raises_on_invalid_quantile() -> None:
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"quantile must lie in \[0, 1\]"):
        bootstrap_quantile(values, quantile=-0.5)
    with pytest.raises(ValueError, match=r"quantile must lie in \[0, 1\]"):
        bootstrap_quantile(values, quantile=1.5)


@pytest.mark.unit
def test_bootstrap_quantile_raises_on_empty_input() -> None:
    empty = np.array([], dtype=float)
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_quantile(empty, quantile=0.5)


@pytest.mark.unit
def test_bootstrap_quantile_raises_on_invalid_alpha() -> None:
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha must lie in"):
        bootstrap_quantile(values, quantile=0.5, alpha=0.0)


@pytest.mark.unit
def test_bootstrap_quantile_raises_on_invalid_n_bootstrap() -> None:
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        bootstrap_quantile(values, quantile=0.5, n_bootstrap=-1)
