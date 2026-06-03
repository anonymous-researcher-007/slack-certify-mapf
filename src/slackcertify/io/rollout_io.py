"""Rollout-result IO via Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slackcertify.simulate.metrics import wilson_ci
from slackcertify.simulate.rollout import RolloutResult

__all__ = ["load_rollout_results", "save_rollout_results"]


def save_rollout_results(results: RolloutResult, path: str | Path) -> None:
    """Write ``results`` to ``path`` as a Parquet file.

    The schema is one row per rollout with the realised collision count
    in the column ``n_collisions``; aggregate fields go into the file
    metadata.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from slackcertify.simulate.rollout import RolloutResult
    >>> r = RolloutResult(
    ...     n_rollouts=3, n_successful=3, success_rate=1.0,
    ...     wilson_ci_95=(0.0, 1.0), mean_executed_soc=1.5,
    ...     mean_executed_makespan=2.0, n_collisions_per_rollout=[0, 0, 0],
    ... )
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "r.parquet"
    >>> save_rollout_results(r, tmp)
    >>> tmp.exists()
    True
    """
    df = pd.DataFrame(
        {
            "rollout": list(range(results.n_rollouts)),
            "n_collisions": list(results.n_collisions_per_rollout),
        }
    )
    df.attrs["n_rollouts"] = results.n_rollouts
    df.attrs["n_successful"] = results.n_successful
    df.attrs["success_rate"] = results.success_rate
    df.attrs["wilson_ci_95_lo"] = results.wilson_ci_95[0]
    df.attrs["wilson_ci_95_hi"] = results.wilson_ci_95[1]
    df.attrs["mean_executed_soc"] = results.mean_executed_soc
    df.attrs["mean_executed_makespan"] = results.mean_executed_makespan
    df.to_parquet(Path(path), index=False)


def load_rollout_results(path: str | Path) -> RolloutResult:
    """Reconstruct a :class:`RolloutResult` from a Parquet file.

    Aggregate fields are recomputed from the per-rollout rows so the
    object is consistent even if the source file was edited.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from slackcertify.simulate.rollout import RolloutResult
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "r.parquet"
    >>> save_rollout_results(RolloutResult(
    ...     n_rollouts=4, n_successful=3, success_rate=0.75,
    ...     wilson_ci_95=(0.0, 1.0), mean_executed_soc=1.0,
    ...     mean_executed_makespan=2.0, n_collisions_per_rollout=[0, 1, 0, 0],
    ... ), tmp)
    >>> r = load_rollout_results(tmp)
    >>> r.n_rollouts == 4 and r.n_successful == 3 and r.success_rate == 0.75
    True
    """
    df = pd.read_parquet(Path(path))
    counts = [int(v) for v in df["n_collisions"].tolist()]
    n = len(counts)
    n_ok = sum(1 for c in counts if c == 0)
    rate = n_ok / n if n else 0.0
    return RolloutResult(
        n_rollouts=n,
        n_successful=n_ok,
        success_rate=rate,
        wilson_ci_95=wilson_ci(n_ok, n) if n else (0.0, 0.0),
        mean_executed_soc=0.0,  # not preserved in per-row schema
        mean_executed_makespan=0.0,
        n_collisions_per_rollout=counts,
    )
