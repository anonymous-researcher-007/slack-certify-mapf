# baselines/ — comparison wrappers for §V

This directory holds wrappers around the *comparison baselines* used in §V of
the ASYU 2026 paper. They live **outside** the `slackcertify/` library
because they are not part of the public API — experiment runners import
them directly to fairly compare Slack-Certify against the field.

| Baseline       | Paper                                       | Axis the comparison isolates                       | Status                                   |
|----------------|---------------------------------------------|----------------------------------------------------|------------------------------------------|
| **kR-CBS**     | Atzmon et al., JAIR 2020 / Chen et al., AAAI 2021 | Bounded-Δ planning *during* solve                | wrapper ready, binary required           |
| **pR-CBS**     | Atzmon et al., ICAPS 2020                   | Probabilistic planning (small instances only)      | wrapper ready, binary required           |
| **BTPG-max**   | Su et al., AAAI 2024 / arXiv:2508.04849 (2025) | Post-hoc TPG relaxation                          | wrapper ready, binary required           |
| **Kottinger**  | arXiv:2307.11252 (2024)                     | Reactive delay introduction (proactive vs. reactive) | wrapper ready, reimplementation stub (no upstream) |

Every wrapper returns a `baselines._types.BaselinePlan(plan, solver_used)`
so the analysis layer can distinguish baselines without inspecting the
:class:`Plan` schema.

## Building the binaries

The kR-CBS, pR-CBS and BTPG wrappers shell out to upstream binaries
expected at `src/slackcertify/solvers/external_bin/<name>`. Build them by
running

```bash
bash scripts/install_baselines.sh
```

(see [`third_party/README.md`](../third_party/README.md) for prerequisites
and pinned commit hashes).

The Kottinger baseline ships with a Python reimplementation in
`baselines/kottinger/_reimpl.py` — no binary needed today; the wrapper
will prefer the upstream binary at
`src/slackcertify/solvers/external_bin/kottinger` once that is released.
