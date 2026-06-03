"""Small statistical helpers shared across the certifier and the simulator.

* :func:`wilson_ci` — Wilson score interval for a binomial proportion.
* :func:`binomial_sf` — survival function for ``Binomial(n, p)``.
* :func:`paired_bootstrap_mean_diff` — paired bootstrap difference of
  means used by §V of the paper for solver-vs-solver comparisons.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, norm

__all__ = ["binomial_sf", "paired_bootstrap_mean_diff", "wilson_ci"]


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return a Wilson score confidence interval for ``k / n``.

    Parameters
    ----------
    k
        Number of "successes".
    n
        Number of trials. Must be positive.
    alpha
        Two-sided significance level (default ``0.05`` → 95 % CI).

    Examples
    --------
    >>> lo, hi = wilson_ci(50, 100)
    >>> 0.3 < lo < 0.5 < hi < 0.7
    True
    >>> wilson_ci(0, 100) == (0.0, wilson_ci(0, 100)[1])
    True
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"k must lie in [0, n], got k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_sf(k: int, n: int, p: float) -> float:
    """Return ``Pr[Binomial(n, p) > k]`` (the survival function).

    Examples
    --------
    >>> round(binomial_sf(0, 1, 0.3), 6)
    0.0
    >>> abs(binomial_sf(0, 1, 0.3) - 0.0) < 1e-9 or binomial_sf(-1, 1, 0.3) > 0
    True
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1], got {p}")
    return float(binom.sf(k, n, p))


def paired_bootstrap_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float]]:
    """Return the paired-bootstrap mean of ``x - y`` and its CI.

    Parameters
    ----------
    x, y
        1-D arrays of paired observations of equal length.
    n_resamples
        Number of bootstrap resamples.
    rng
        Random generator (defaults to a fresh ``np.random.default_rng``).
    alpha
        Two-sided significance level.

    Returns
    -------
    (point_estimate, (ci_lo, ci_hi))
        ``point_estimate`` is ``mean(x - y)``; the CI is the
        ``[alpha/2, 1 - alpha/2]`` percentiles of the bootstrap
        resample distribution.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([0.0, 1.0, 2.0])
    >>> point, ci = paired_bootstrap_mean_diff(x, y, n_resamples=500,
    ...                                        rng=np.random.default_rng(0))
    >>> abs(point - 1.0) < 1e-9 and ci[0] <= 1.0 <= ci[1]
    True
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError("x and y must be non-empty")
    if rng is None:
        rng = np.random.default_rng()
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = diffs.size
    sample_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(low=0, high=n, size=n)
        sample_means[i] = float(np.mean(diffs[idx]))
    lo = float(np.quantile(sample_means, alpha / 2.0))
    hi = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return float(np.mean(diffs)), (lo, hi)
