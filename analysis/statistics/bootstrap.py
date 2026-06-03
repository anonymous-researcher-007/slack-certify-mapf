"""Bootstrap CIs for paired-mean differences and quantiles."""

from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_quantile", "paired_bootstrap_mean_diff"]


def paired_bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return ``(mean_diff, ci_lo, ci_hi)`` for the paired difference ``a - b``.

    Resamples paired ``(a_i, b_i)`` indices with replacement
    ``n_bootstrap`` times and reads off the empirical
    ``[alpha/2, 1-alpha/2]`` quantiles of the resampled mean-difference
    distribution.

    Parameters
    ----------
    a, b
        1-D NumPy arrays of equal length.
    n_bootstrap
        Resample count. Defaults to 10 000.
    alpha
        Two-sided significance level (0.05 → 95 % CI).
    rng
        Source of randomness. Defaults to ``np.random.default_rng()``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([3.0, 4.0, 5.0])
    >>> b = np.array([1.0, 2.0, 3.0])
    >>> point, lo, hi = paired_bootstrap_mean_diff(
    ...     a, b, n_bootstrap=500, rng=np.random.default_rng(0)
    ... )
    >>> abs(point - 2.0) < 1e-9 and lo <= 2.0 <= hi
    True
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"a and b must have the same shape, got {a_arr.shape} and {b_arr.shape}")
    if a_arr.size == 0:
        raise ValueError("a and b must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    if rng is None:
        rng = np.random.default_rng()
    diffs = a_arr - b_arr
    n = diffs.size
    idx = rng.integers(low=0, high=n, size=(n_bootstrap, n))
    sample_means = diffs[idx].mean(axis=1)
    lo = float(np.quantile(sample_means, alpha / 2.0))
    hi = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return float(diffs.mean()), lo, hi


def bootstrap_quantile(
    values: np.ndarray,
    quantile: float,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, ci_lo, ci_hi)`` for a sample quantile.

    Examples
    --------
    >>> import numpy as np
    >>> point, lo, hi = bootstrap_quantile(
    ...     np.arange(100, dtype=float),
    ...     quantile=0.5,
    ...     n_bootstrap=500,
    ...     rng=np.random.default_rng(0),
    ... )
    >>> 40 < lo <= point <= hi < 60
    True
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must lie in [0, 1], got {quantile}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    if rng is None:
        rng = np.random.default_rng()
    point = float(np.quantile(arr, quantile))
    n = arr.size
    idx = rng.integers(low=0, high=n, size=(n_bootstrap, n))
    sample_quantiles = np.quantile(arr[idx], quantile, axis=1)
    lo = float(np.quantile(sample_quantiles, alpha / 2.0))
    hi = float(np.quantile(sample_quantiles, 1.0 - alpha / 2.0))
    return point, lo, hi
