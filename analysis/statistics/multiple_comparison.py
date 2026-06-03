"""Multiple-comparison corrections — Holm-Bonferroni and Benjamini-Hochberg."""

from __future__ import annotations

import numpy as np

__all__ = ["benjamini_hochberg", "holm_bonferroni"]


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return a boolean reject-mask under Holm-Bonferroni at family-wise ``alpha``.

    The Holm step-down procedure sorts p-values in ascending order and
    rejects ``H_(k)`` iff ``p_(k) <= alpha / (m - k + 1)`` for all
    ``k' <= k``. The first failure stops the rejection chain.

    Parameters
    ----------
    p_values
        One p-value per hypothesis, in any order.
    alpha
        Family-wise significance level.

    Returns
    -------
    list of bool
        ``True`` at index ``i`` iff hypothesis ``i`` is rejected.
        Returned in the input order.

    Examples
    --------
    >>> holm_bonferroni([0.001, 0.04, 0.03, 0.5], alpha=0.05)
    [True, False, False, False]
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    m = len(p_values)
    if m == 0:
        return []
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("every p-value must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    rejected = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return [bool(x) for x in rejected]


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return a boolean reject-mask under Benjamini-Hochberg FDR control.

    Sorts p-values ascending; finds the largest ``k`` with
    ``p_(k) <= k / m * alpha``; rejects ``H_(1) ... H_(k)``.

    Examples
    --------
    >>> benjamini_hochberg([0.001, 0.02, 0.03, 0.04], alpha=0.05)
    [True, True, True, True]
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    m = len(p_values)
    if m == 0:
        return []
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("every p-value must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return [False] * m
    k_max = int(np.max(np.where(below)[0]))
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return [bool(x) for x in rejected]
