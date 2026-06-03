# kR-CBS / kR-PBS

**k-Robust CBS** and **k-Robust PBS** are robust extensions of CBS / PBS that
plan trajectories which are guaranteed conflict-free under any sequence of
delays whose cumulative magnitude per agent is at most ``k``. They are the
natural "fix it at solve time" baseline for Slack-Certify's "fix it
afterwards" approach in the bounded-Δ regime of §V.

* **kR-CBS** — Atzmon et al., *Robust Multi-Agent Path Finding and Executing*,
  JAIR 67 (2020).
* **kR-EECBS / kR-PBS** — Chen et al., *k-Robust Multi-Agent Path Finding*,
  AAAI 2021. We wrap their public release.

## Binaries

| Wrapper                      | Binary path (built by `scripts/install_baselines.sh`)        |
|------------------------------|--------------------------------------------------------------|
| `KRobustCBSBaseline`         | `src/slackcertify/solvers/external_bin/kr_eecbs`             |
| `KRobustPBSBaseline`         | `src/slackcertify/solvers/external_bin/kr_pbs`               |

## Provenance tag

Each call to :meth:`solve` returns a
:class:`baselines._types.BaselinePlan` with
``solver_used = f"kr_eecbs_k={k}_w={suboptimality}"`` (or the matching
``kr_pbs_…`` form). Analysis aggregations group by this string.

## Patches

Any small patches we need to apply to the upstream tree before building
live in `baselines/kr_cbs/patches/` (currently empty). The
`scripts/install_baselines.sh` driver does not yet apply them
automatically — invoke `git apply` manually if needed.
