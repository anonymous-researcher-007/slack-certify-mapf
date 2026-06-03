"""Unit tests for the analysis statistics helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from analysis.statistics.bootstrap import (
    bootstrap_quantile,
    paired_bootstrap_mean_diff,
)
from analysis.statistics.multiple_comparison import (
    benjamini_hochberg,
    holm_bonferroni,
)
from analysis.statistics.wilson import wilson_ci, wilson_ci_dataframe


@pytest.mark.unit
def test_wilson_ci_on_known_inputs() -> None:
    """k=50, n=100, alpha=0.05 should give a 95% CI roughly (0.40, 0.60)."""
    lo, hi = wilson_ci(50, 100, alpha=0.05)
    # The exact Wilson CI for 50/100 at 95% is approximately
    # (0.4038, 0.5962); pin to a ±0.005 band around those values.
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61
    assert lo < 0.5 < hi


@pytest.mark.unit
def test_wilson_ci_dataframe_adds_columns() -> None:
    df = pd.DataFrame({"k": [0, 50, 100], "n": [100, 100, 100]})
    out = wilson_ci_dataframe(df, "k", "n")
    assert {"ci_lo", "ci_hi"}.issubset(out.columns)
    # Extreme cases (k=0 and k=n) clamp the CI to [0, ·] / [·, 1] up to
    # floating-point noise from the closed-form Wilson formula.
    assert out.loc[0, "ci_lo"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[2, "ci_hi"] == pytest.approx(1.0, abs=1e-9)
    # k=50/100 sits inside its own CI.
    assert out.loc[1, "ci_lo"] < 0.5 < out.loc[1, "ci_hi"]


@pytest.mark.unit
def test_paired_bootstrap_on_known_distribution() -> None:
    """Constant-difference paired arrays must return that constant as the point estimate."""
    rng = np.random.default_rng(0)
    a = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    point, lo, hi = paired_bootstrap_mean_diff(a, b, n_bootstrap=2000, rng=rng)
    # Diffs are exactly 2.0 each, so every bootstrap sample is 2.0.
    assert math.isclose(point, 2.0, abs_tol=1e-9)
    assert math.isclose(lo, 2.0, abs_tol=1e-9)
    assert math.isclose(hi, 2.0, abs_tol=1e-9)


@pytest.mark.unit
def test_paired_bootstrap_ci_covers_truth_for_random_input() -> None:
    """With genuine variance the CI should cover the true mean difference."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=2.0, scale=1.0, size=200)
    b = rng.normal(loc=0.0, scale=1.0, size=200)
    point, lo, hi = paired_bootstrap_mean_diff(a, b, n_bootstrap=2000, rng=rng)
    assert lo <= point <= hi
    # True mean difference is 2.0; the CI should contain it.
    assert lo <= 2.0 <= hi


@pytest.mark.unit
def test_bootstrap_quantile_median_of_uniform() -> None:
    rng = np.random.default_rng(7)
    values = np.arange(100, dtype=float)
    point, lo, hi = bootstrap_quantile(values, quantile=0.5, n_bootstrap=2000, rng=rng)
    # Sample median of 0..99 is 49.5; bootstrap CI must cover the point.
    assert math.isclose(point, 49.5, abs_tol=1e-9)
    assert lo <= point <= hi
    # And the bootstrap CI must lie inside the empirical support [0, 99].
    assert 30.0 < lo < hi < 70.0


@pytest.mark.unit
def test_holm_bonferroni_on_known_p_values() -> None:
    # Classic Holm example: only the strongest p-value passes the
    # stepped-down threshold at alpha=0.05.
    rejects = holm_bonferroni([0.001, 0.04, 0.03, 0.5], alpha=0.05)
    assert rejects == [True, False, False, False]


@pytest.mark.unit
def test_holm_bonferroni_all_significant_when_well_below_alpha() -> None:
    rejects = holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
    assert rejects == [True, True, True]


@pytest.mark.unit
def test_holm_bonferroni_empty_input() -> None:
    assert holm_bonferroni([], alpha=0.05) == []


@pytest.mark.unit
def test_benjamini_hochberg_known_p_values() -> None:
    # Sorted p-values [0.001, 0.02, 0.03, 0.04] vs BH thresholds
    # [0.0125, 0.025, 0.0375, 0.05]: all four pass i/m * alpha, so BH
    # rejects every hypothesis. Holm on the same input stops at rank 2
    # because 0.02 > 0.05/3 = 0.0167; this contrast pins BH as the
    # less-conservative procedure.
    rejects = benjamini_hochberg([0.001, 0.02, 0.03, 0.04], alpha=0.05)
    assert rejects == [True, True, True, True]


@pytest.mark.unit
def test_benjamini_hochberg_no_rejections_when_all_p_large() -> None:
    rejects = benjamini_hochberg([0.5, 0.6, 0.7], alpha=0.05)
    assert rejects == [False, False, False]


@pytest.mark.unit
def test_benjamini_hochberg_empty_input() -> None:
    assert benjamini_hochberg([], alpha=0.05) == []
