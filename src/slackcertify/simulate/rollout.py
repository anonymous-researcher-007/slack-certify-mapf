"""Monte-Carlo rollout driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from joblib import Parallel, delayed

from slackcertify.core.plan import Plan
from slackcertify.simulate.executor import Executor
from slackcertify.simulate.metrics import wilson_ci
from slackcertify.utils.seed import derive_seed

__all__ = ["DelayModel", "RolloutResult", "monte_carlo_rollout"]


class DelayModel(Protocol):
    """Structural protocol matching every model in :mod:`slackcertify.delay`."""

    def sample(  # noqa: D401 - protocol spec
        self, plan: Plan, rng: np.random.Generator
    ) -> dict[int, list[int]]:
        """Sample a per-agent integer delay schedule for ``plan``."""
        ...


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Aggregate statistics of a Monte-Carlo rollout study.

    Attributes
    ----------
    n_rollouts
        Number of independent rollouts executed.
    n_successful
        Rollouts that completed without any collision.
    success_rate
        ``n_successful / n_rollouts``.
    wilson_ci_95
        95 % Wilson score CI on ``success_rate``.
    mean_executed_soc
        Mean realised sum-of-costs across rollouts (sum of per-agent
        wall-clock arrival times).
    mean_executed_makespan
        Mean realised makespan (max wall-clock arrival) across rollouts.
    n_collisions_per_rollout
        Per-rollout collision count.
    """

    n_rollouts: int
    n_successful: int
    success_rate: float
    wilson_ci_95: tuple[float, float]
    mean_executed_soc: float
    mean_executed_makespan: float
    n_collisions_per_rollout: list[int]


def monte_carlo_rollout(
    plan: Plan,
    delay_model: DelayModel,
    K: int = 500,  # noqa: N803 - public API uses paper notation
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> RolloutResult:
    """Run ``K`` independent rollouts of ``plan`` under ``delay_model``.

    Per-rollout seeds are derived deterministically from a master seed
    drawn from ``rng``, so results are reproducible across the
    ``n_jobs`` parameter (sequential vs. parallel produces identical
    aggregate statistics).

    Parameters
    ----------
    plan
        The plan to execute.
    delay_model
        Any object exposing ``sample(plan, rng)``.
    K
        Number of rollouts.
    rng
        Source of the master seed. Defaults to a fresh
        ``np.random.default_rng()``.
    n_jobs
        Joblib parallelism level; ``-1`` uses every core. Defaults to
        ``1`` (sequential) which is fastest for small ``K``.

    Examples
    --------
    >>> import numpy as np
    >>> from slackcertify.core.plan import Agent, Path, Plan
    >>> from slackcertify.delay.bernoulli import BernoulliDelayModel
    >>> a = [Agent(id=0, start=(0, 0), goal=(2, 0))]
    >>> p = [Path(agent_id=0, vertices=[(0, 0), (1, 0), (2, 0)])]
    >>> r = monte_carlo_rollout(
    ...     Plan.from_paths(a, p),
    ...     BernoulliDelayModel(p_d=0.1),
    ...     K=20,
    ...     rng=np.random.default_rng(0),
    ... )
    >>> r.n_rollouts == 20 and r.success_rate == 1.0
    True
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if rng is None:
        rng = np.random.default_rng()
    base_seed = int(rng.integers(0, 2**32 - 1))

    def _one(k: int) -> tuple[bool, int, int, int]:
        """Run a single rollout ``k`` and return ``(success, n_collisions, soc, makespan)``."""
        worker_rng = np.random.default_rng(derive_seed(base_seed, f"rollout-{k}"))
        delays = delay_model.sample(plan, worker_rng)
        trace = Executor(plan, delays).run()
        executed_soc = sum(trace.realised_arrival.values())
        executed_makespan = max(trace.realised_arrival.values(), default=0)
        return trace.success, len(trace.collisions), executed_soc, executed_makespan

    if n_jobs == 1:
        results = [_one(k) for k in range(K)]
    else:
        raw: Any = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(k) for k in range(K))
        results = list(raw)

    successes = [r[0] for r in results]
    coll_counts = [r[1] for r in results]
    socs = [r[2] for r in results]
    makespans = [r[3] for r in results]

    n_successful = int(sum(successes))
    success_rate = n_successful / K
    return RolloutResult(
        n_rollouts=K,
        n_successful=n_successful,
        success_rate=success_rate,
        wilson_ci_95=wilson_ci(n_successful, K),
        mean_executed_soc=float(np.mean(socs)) if socs else 0.0,
        mean_executed_makespan=float(np.mean(makespans)) if makespans else 0.0,
        n_collisions_per_rollout=coll_counts,
    )
