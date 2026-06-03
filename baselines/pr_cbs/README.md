# pR-CBS

**p-Robust CBS** is the probabilistic-robustness extension of CBS:

* Atzmon et al., *Probabilistic Robust Multi-Agent Path Finding*,
  ICAPS 2020.

The published scaling limit is roughly **8 agents on 8×8 empty grids** —
the joint-Bayesian inference inside the CBS high-level node blows up
combinatorially beyond that. We therefore use pR-CBS only as a **§V RQ4
small-instance baseline**; for everything else it would dominate the
runtime budget.

## Binary

| Wrapper                | Binary path                                                     |
|------------------------|-----------------------------------------------------------------|
| `PRobustCBSBaseline`   | `src/slackcertify/solvers/external_bin/pr_cbs`                  |

## Provenance tag

Each call to :meth:`solve` returns a
:class:`baselines._types.BaselinePlan` with
``solver_used = f"pr_cbs_p={p_robust}_pd={p_d}"``.

The wrapper raises :class:`SolverTimeoutError` *gracefully*: callers are
expected to treat timeouts as expected when the instance exceeds the
~8-agent scaling threshold.
