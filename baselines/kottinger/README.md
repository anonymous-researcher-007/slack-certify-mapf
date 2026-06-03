# Kottinger CBS-delay

**Kottinger CBS-delay** is a *reactive* delay-introduction baseline:
when an execution-time delay event is observed, it inserts the
minimum number of waits to restore Δ-disjointness with respect to the
post-event state.

* arXiv:2307.11252 (2024).
* Upstream source: <https://github.com/aria-systems-group/Delay-Robust-MAPF>.

## Why offline?

For §V we run Kottinger **offline** against worst-case Δ so the
proactive-vs-reactive axis is the *sole* differentiator from
Slack-Certify. The reactive-online setting is included for
completeness via :meth:`solve_online`, but it is not on the §V
critical path.

## Implementation status

The wrapper invokes the **upstream Kottinger et al. 2024 binary**
(`aria-systems-group/Delay-Robust-MAPF`), built by
`scripts/install_baselines.sh` and installed at
`src/slackcertify/solvers/external_bin/kottinger`. The wrapper
auto-prefers the binary when present. When the binary is absent
(e.g. in CI sandboxes that can't build the C++ source) the wrapper
falls back to the in-process reimplementation in
`baselines/kottinger/_reimpl.py`. The fallback delegates to the
greedy `slack_certify` certifier; its output schema (a Δ-disjoint
:class:`Plan`) matches the upstream's, but the greedy waits may
exceed the CBS-style optimum on hard instances. Paper-grade
comparisons therefore require the real binary; the fallback exists
to keep the test pipeline end-to-end runnable in sandboxes.

The min-delay-introduction problem is **APX-hard** per
Kottinger 2024; both the upstream and the fallback honour the
caller-supplied wall-clock budget.

## Forcing the binary path

`KottingerDelayBaseline._use_binary(baseline)` returns `True` iff
the binary exists at the expected path. Callers that want to fail
loudly when the binary is missing (rather than silently fall back to
the reimpl) can check `_use_binary` themselves and route around the
wrapper.

## Provenance tag

Returned :class:`baselines._types.BaselinePlan` carry
`solver_used = f"<nominal>+kottinger_offline_delta={delta}_{provenance}"`,
where `{provenance}` is `binary` (upstream binary executed) or
`reimpl` (sandboxed fallback). The analysis pipeline filters on
this suffix when reporting which results came from the upstream
reference implementation versus the in-process approximation.
