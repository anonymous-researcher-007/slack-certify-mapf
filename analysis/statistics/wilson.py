"""Wilson confidence intervals for binomial proportions.

Thin re-export and DataFrame-friendly wrapper around
:func:`slackcertify.utils.stats.wilson_ci`. Kept here so the analysis
pipeline doesn't depend on any internal module path.
"""

from __future__ import annotations

import pandas as pd

from slackcertify.utils.stats import wilson_ci as _wilson_ci

__all__ = ["wilson_ci", "wilson_ci_dataframe"]


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return a Wilson score CI for ``k / n`` at significance ``alpha``.

    Forwarding wrapper around :func:`slackcertify.utils.stats.wilson_ci`;
    isolates the analysis layer from the internal module path so future
    refactors don't ripple into the plots.

    Examples
    --------
    >>> lo, hi = wilson_ci(50, 100)
    >>> 0.3 < lo < 0.5 < hi < 0.7
    True
    """
    return _wilson_ci(k, n, alpha=alpha)


def wilson_ci_dataframe(
    df: pd.DataFrame,
    k_col: str,
    n_col: str,
    alpha: float = 0.05,
    lo_col: str = "ci_lo",
    hi_col: str = "ci_hi",
) -> pd.DataFrame:
    """Return a copy of ``df`` with two extra Wilson-CI columns.

    Rows where ``df[n_col] <= 0`` produce ``NaN`` bounds rather than
    raising. ``k`` is clamped into ``[0, n]`` defensively.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"k": [50, 90], "n": [100, 100]})
    >>> out = wilson_ci_dataframe(df, "k", "n")
    >>> bool((out["ci_lo"] < out["k"] / out["n"]).all())
    True
    """
    out = df.copy()
    lo_vals: list[float] = []
    hi_vals: list[float] = []
    for _, row in out.iterrows():
        n_val = int(row[n_col])
        k_val = int(row[k_col])
        if n_val <= 0:
            lo_vals.append(float("nan"))
            hi_vals.append(float("nan"))
            continue
        k_clamped = max(0, min(n_val, k_val))
        lo, hi = _wilson_ci(k_clamped, n_val, alpha=alpha)
        lo_vals.append(lo)
        hi_vals.append(hi)
    out[lo_col] = lo_vals
    out[hi_col] = hi_vals
    return out
