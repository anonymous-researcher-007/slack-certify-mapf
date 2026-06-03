# BTPG-max

**BTPG (Bidirectional Temporal Plan Graph) max** is a post-hoc *offline*
relaxation layer for an already-solved nominal MAPF plan: it consumes
the plan's TPG and selectively *removes* Type-2 edges so the resulting
schedule has more execution slack, while preserving feasibility.

* Su et al., AAAI 2024 / arXiv:2508.04849 (2025).

## Binary

| Wrapper            | Binary path                                                   |
|--------------------|---------------------------------------------------------------|
| `BTPGMaxBaseline`  | `src/slackcertify/solvers/external_bin/btpg`                  |

## Provenance tag

The wrapper does not own a `solve()` method — BTPG is *post-hoc*, not a
solver. Use :meth:`BTPGMaxBaseline.post_process(plan, time_limit_s)`,
which returns a :class:`baselines._types.BaselinePlan` whose
``solver_used`` is ``"<nominal>+btpg_max"``.
