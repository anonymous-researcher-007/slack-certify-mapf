---
name: Bug report
about: Report a defect in the certifier, runner, analysis, or CI.
title: "[BUG] "
labels: ["bug"]
assignees: []
---

## Description

<!-- One-paragraph summary of the bug. -->

## Reproduction

<!-- Minimal command(s) or code snippet that triggers the bug. Prefer
a snippet that runs against tests/data/ rather than benchmarks/, so
maintainers can reproduce without downloading the full MovingAI
benchmark suite. -->

```bash
# e.g.:
pytest tests/integration/test_rq1_smoke.py -v
```

## Expected behaviour

<!-- What should have happened? -->

## Actual behaviour

<!-- What actually happened? Paste the error message / traceback / log
output verbatim. -->

```
<paste here>
```

## Environment

- OS: <!-- e.g., Ubuntu 22.04, macOS 14.4 -->
- Python version (from `python --version`):
- `slack-certify-mapf` version (from `pip show slackcertify` or commit hash):
- Output of `python scripts/check_environment.py` (paste below):

```
<paste here>
```

## Additional context

<!-- Anything else: hypotheses about the cause, related issues, etc. -->
